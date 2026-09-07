package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"math/big"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

// A prior success qualifies only the same complete requested configuration.
// Measurement controls are included here even though trace launch later forces
// the warmer off; eligibility refers to the requested timing configuration.
type traceConfigKey struct {
	Tier         int
	Suite        string
	Benchmark    string
	ScalingName  string
	ScalingValue string
	NonScaling   string
}

type traceSource struct {
	RunID     int64
	SessionID int64
}

type successfulConfigs map[traceConfigKey]traceSource

// canonicalConfigParams ignores JSON key order, whitespace, and equivalent
// number spellings while preserving scalar types and exact numeric values.
// In particular, large integer values must not alias through float64 rounding.
func canonicalConfigParams(raw string) (string, error) {
	var params map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &params); err != nil {
		return "", err
	}
	if params == nil {
		return "", fmt.Errorf("non-scaling parameters must be a JSON object")
	}
	canonical := make(map[string]string, len(params))
	for name, rawValue := range params {
		var value interface{}
		decoder := json.NewDecoder(strings.NewReader(string(rawValue)))
		decoder.UseNumber()
		if err := decoder.Decode(&value); err != nil {
			return "", err
		}
		switch v := value.(type) {
		case json.Number:
			number, ok := new(big.Rat).SetString(string(v))
			if !ok {
				return "", fmt.Errorf("parameter %q has an unsupported number", name)
			}
			canonical[name] = "number:" + number.RatString()
		case string:
			canonical[name] = "string:" + v
		case bool:
			canonical[name] = fmt.Sprintf("bool:%t", v)
		default:
			return "", fmt.Errorf("parameter %q must have a scalar value", name)
		}
	}
	encoded, err := json.Marshal(canonical)
	return string(encoded), err
}

func requestedTraceConfig(b *BenchmarkParams, cfg configPair) (traceConfigKey, error) {
	raw, err := json.Marshal(cfg.NSDict)
	if err != nil {
		return traceConfigKey{}, err
	}
	ns, err := canonicalConfigParams(string(raw))
	return traceConfigKey{b.Tier, b.Suite, b.Benchmark, b.ScalingParam.Name,
		formatValue(cfg.ScalingValue), ns}, err
}

// The selected DB is the eligibility scope, including all of its sessions.
// Keep the first successful source row deterministically for diagnostics.
func loadSuccessfulConfigs(db *sql.DB) (successfulConfigs, error) {
	rows, err := db.Query(`SELECT r.id, r.session_id, r.tier, r.suite,
		r.benchmark, r.scaling_param_name, r.scaling_param_value,
		r.non_scaling_params_json
		FROM runs r JOIN sessions s ON s.id = r.session_id
		WHERE r.status = 'ok' ORDER BY r.id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	result := make(successfulConfigs)
	for rows.Next() {
		var key traceConfigKey
		var source traceSource
		var raw string
		if err := rows.Scan(&source.RunID, &source.SessionID, &key.Tier,
			&key.Suite, &key.Benchmark, &key.ScalingName, &key.ScalingValue, &raw); err != nil {
			return nil, err
		}
		key.NonScaling, err = canonicalConfigParams(raw)
		if err != nil {
			return nil, fmt.Errorf("run %d: %w", source.RunID, err)
		}
		if _, exists := result[key]; !exists {
			result[key] = source
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

// A missing DB preserves unfiltered tracing. An existing but unreadable or
// invalid DB must not silently turn failed guidance into permission to trace.
func loadTraceGuidance(dbPath string) (successfulConfigs, error) {
	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	absolute, err := filepath.Abs(dbPath)
	if err != nil {
		return nil, err
	}
	dsn := (&url.URL{Scheme: "file", Path: absolute, RawQuery: "mode=ro"}).String()
	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return nil, err
	}
	defer db.Close()
	return loadSuccessfulConfigs(db)
}
