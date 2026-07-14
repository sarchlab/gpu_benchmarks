package main

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// KernelResult represents a parsed kernel event. Benchmarks that loop the
// measurement in-program emit one kernel event per timed iteration, carrying
// the iteration index and the host-side wall-clock time alongside the GPU
// event time.
type KernelResult struct {
	Name       string   `json:"name"`
	TimeMS     float64  `json:"time_ms"`
	HostTimeMS *float64 `json:"host_time_ms"`
	Iteration  *int     `json:"iter"`
	Ordinal    int      `json:"-"`
	// Params is the raw JSON of the event's "params" object (the actual
	// parameter values the benchmark used), stored verbatim for the
	// parameter-verification contract in docs/output_protocol.md. Empty if the
	// event omitted it.
	Params string `json:"-"`
	// ParamsMap is the decoded params object, used by validateRun to check the
	// reported values against the requested configuration. Nil when the event
	// omitted params or params was not a JSON object.
	ParamsMap map[string]interface{} `json:"-"`
	// HasParams records whether the required "params" object was present.
	HasParams bool `json:"-"`
	// HasTime records whether the required "time_ms" field was present, so the
	// harness can distinguish a real 0 ms from a missing field.
	HasTime bool `json:"-"`
}

// SummaryResult represents a parsed summary event.
type SummaryResult struct {
	TotalTimeMS *float64       `json:"total_time_ms"`
	Metrics     []MetricResult `json:"metrics"`
}

// MetricResult represents a metric within a summary.
type MetricResult struct {
	Name  string  `json:"name"`
	Value float64 `json:"value"`
}

// ErrorResult represents a parsed error event. Severity is "fatal" or
// "warning" per docs/output_protocol.md; an empty severity is treated as fatal.
type ErrorResult struct {
	Message  string `json:"message"`
	Severity string `json:"severity"`
}

// IsWarning reports whether the error event is a non-fatal warning.
func (e ErrorResult) IsWarning() bool {
	return e.Severity == "warning"
}

// ParsedOutput is everything parseOutput extracted from a run's stdout.
// Violations records line-level protocol breaks (non-JSON lines, unknown event
// types, events missing required fields) so the run-status decision can reject
// malformed output instead of silently dropping it.
type ParsedOutput struct {
	Kernels      []KernelResult
	Summary      *SummaryResult // last summary event seen (kept for storage)
	SummaryCount int
	Errors       []ErrorResult
	Violations   []string
}

// parseOutput parses JSON-lines stdout per docs/output_protocol.md. Malformed
// lines and events are still parsed as far as possible (so partial data can be
// stored for diagnosis) but are recorded as protocol violations.
func parseOutput(stdoutData string) ParsedOutput {
	var out ParsedOutput
	ordinal := 0

	for _, line := range strings.Split(stdoutData, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(line), &obj); err != nil {
			// stdout is structured-data-only (docs/output_protocol.md): any
			// non-empty line that isn't valid JSON is unexpected output or a
			// lost/truncated record.
			preview := line
			if len(preview) > 120 {
				preview = preview[:120] + "..."
			}
			out.Violations = append(out.Violations,
				fmt.Sprintf("non-JSON stdout line: %s", preview))
			continue
		}

		typeStr, _ := obj["type"].(string)
		switch typeStr {
		case "kernel":
			name, _ := obj["name"].(string)
			if name == "" {
				out.Violations = append(out.Violations,
					"kernel event missing required name")
				name = "unknown"
			}
			timeMS := 0.0
			hasTime := false
			if t, ok := obj["time_ms"].(float64); ok {
				timeMS = t
				hasTime = true
			} else {
				// time_ms is a required field; its absence means a malformed
				// event whose stored 0.0 would masquerade as a real 0 ms.
				out.Violations = append(out.Violations,
					fmt.Sprintf("kernel event %q missing required time_ms", name))
			}
			kr := KernelResult{
				Name:    name,
				TimeMS:  timeMS,
				Ordinal: ordinal,
				HasTime: hasTime,
			}
			if h, ok := obj["host_time_ms"].(float64); ok {
				kr.HostTimeMS = &h
			}
			if it, ok := obj["iter"].(float64); ok {
				i := int(it)
				kr.Iteration = &i
			}
			if p, ok := obj["params"]; ok {
				if pMap, ok := p.(map[string]interface{}); ok {
					kr.HasParams = true
					kr.ParamsMap = pMap
					if b, err := json.Marshal(p); err == nil {
						kr.Params = string(b)
					}
				} else {
					out.Violations = append(out.Violations,
						fmt.Sprintf("kernel event %q params is not a JSON object", name))
				}
			}
			out.Kernels = append(out.Kernels, kr)
			ordinal++

		case "summary":
			out.SummaryCount++
			s := &SummaryResult{}
			if t, ok := obj["total_time_ms"].(float64); ok {
				s.TotalTimeMS = &t
			}
			if metricsRaw, ok := obj["metrics"].([]interface{}); ok {
				for _, m := range metricsRaw {
					mObj, ok := m.(map[string]interface{})
					if !ok {
						out.Violations = append(out.Violations,
							"summary metric entry is not a JSON object")
						continue
					}
					name, _ := mObj["name"].(string)
					if name == "" {
						out.Violations = append(out.Violations,
							"summary metric missing required name")
						name = "unknown"
					}
					value := 0.0
					if v, ok := mObj["value"].(float64); ok {
						value = v
					} else {
						out.Violations = append(out.Violations,
							fmt.Sprintf("summary metric %q missing required numeric value", name))
					}
					s.Metrics = append(s.Metrics, MetricResult{
						Name:  name,
						Value: value,
					})
				}
			}
			out.Summary = s

		case "error":
			msg, _ := obj["message"].(string)
			sev, _ := obj["severity"].(string)
			if msg == "" {
				out.Violations = append(out.Violations,
					"error event missing required message")
			}
			out.Errors = append(out.Errors, ErrorResult{Message: msg, Severity: sev})

		case "info":
			// Recognized by the protocol; carries free-form metadata that is
			// not currently stored (see audit finding E-06).

		default:
			out.Violations = append(out.Violations,
				fmt.Sprintf("unknown event type %q", typeStr))
		}
	}

	return out
}

// expectedRunParams builds the parameter set every kernel event must echo:
// the scaling param plus all non-scaling params, minus harness-level
// measurement controls (e.g. vector_add_warmup), which benchmarks do not
// report in their params objects.
func expectedRunParams(scalingName string, scalingValue interface{},
	nsDict map[string]interface{}) map[string]interface{} {
	expected := map[string]interface{}{scalingName: scalingValue}
	for k, v := range nsDict {
		if !measurementParams[k] {
			expected[k] = v
		}
	}
	return expected
}

// paramValueEq compares a requested parameter value against the value a kernel
// event reported. Numbers compare numerically (a numeric string on either side
// is coerced) with a tiny relative tolerance for decimal round-tripping;
// everything else compares by canonical string form.
func paramValueEq(want, got interface{}) bool {
	wf, wok := paramAsFloat(want)
	gf, gok := paramAsFloat(got)
	if wok && gok {
		if wf == gf {
			return true
		}
		scale := math.Max(math.Abs(wf), math.Abs(gf))
		return math.Abs(wf-gf) <= 1e-9*scale
	}
	return formatValue(want) == formatValue(got)
}

func paramAsFloat(v interface{}) (float64, bool) {
	if f := toFloat(v); f != nil {
		return *f, true
	}
	if s, ok := v.(string); ok {
		if f, err := strconv.ParseFloat(s, 64); err == nil {
			return f, true
		}
	}
	return 0, false
}

// summaryTolerance is the allowed |total_time_ms - Σ kernel time_ms| slack:
// an absolute floor for %.6f decimal round-tripping of each emitted record
// plus a small relative term for accumulation order.
func summaryTolerance(sum float64, nKernels int) float64 {
	return 1e-3 + 1e-6*math.Abs(sum) + 1e-6*float64(nKernels)
}

// validateRun checks an exit-zero run against the full output protocol
// (docs/output_protocol.md, methodology rules 7-8): exactly one summary with a
// finite non-negative total_time_ms that matches the sum of kernel times, at
// least one kernel event, finite non-negative times, only declared kernel
// names, params present on every kernel event and equal to the requested
// configuration, no fatal error event, and no malformed or unknown stdout
// lines. Returns the list of violations; empty means the run is acceptable.
func validateRun(out ParsedOutput, declaredKernels []string,
	expected map[string]interface{}) []string {
	violations := append([]string{}, out.Violations...)
	add := func(format string, args ...interface{}) {
		violations = append(violations, fmt.Sprintf(format, args...))
	}

	// A fatal error event with exit code 0 is a contradiction: the benchmark
	// reported it could not produce valid results.
	for _, e := range out.Errors {
		if !e.IsWarning() {
			add("fatal error event reported despite exit code 0: %s", e.Message)
			break
		}
	}

	switch {
	case out.SummaryCount == 0:
		add("no summary event")
	case out.SummaryCount > 1:
		add("%d summary events (want exactly 1)", out.SummaryCount)
	}
	if out.SummaryCount > 0 && (out.Summary == nil || out.Summary.TotalTimeMS == nil) {
		add("summary missing required numeric total_time_ms")
	}

	if len(out.Kernels) == 0 {
		add("no kernel events")
	}

	declared := make(map[string]bool, len(declaredKernels))
	for _, k := range declaredKernels {
		declared[k] = true
	}

	// Dedupe per-kernel violations: per-iteration benchmarks repeat the same
	// defect on every event, and one report per distinct problem is enough.
	seen := map[string]bool{}
	addOnce := func(format string, args ...interface{}) {
		msg := fmt.Sprintf(format, args...)
		if !seen[msg] {
			seen[msg] = true
			violations = append(violations, msg)
		}
	}

	timesValid := true
	kernelSum := 0.0
	for _, kr := range out.Kernels {
		if kr.HasTime {
			if math.IsNaN(kr.TimeMS) || math.IsInf(kr.TimeMS, 0) || kr.TimeMS < 0 {
				addOnce("kernel %q has invalid time_ms %v", kr.Name, kr.TimeMS)
				timesValid = false
			} else {
				kernelSum += kr.TimeMS
			}
		} else {
			// Already recorded as a parse violation; the sum is unusable.
			timesValid = false
		}
		if kr.HostTimeMS != nil {
			h := *kr.HostTimeMS
			if math.IsNaN(h) || math.IsInf(h, 0) || h < 0 {
				addOnce("kernel %q has invalid host_time_ms %v", kr.Name, h)
			}
		}
		if len(declared) > 0 && !declared[kr.Name] {
			addOnce("kernel name %q not in declared kernels list", kr.Name)
		}
		if !kr.HasParams {
			addOnce("kernel %q missing required params object", kr.Name)
			continue
		}
		for name, want := range expected {
			got, ok := kr.ParamsMap[name]
			if !ok {
				addOnce("kernel %q params missing %q", kr.Name, name)
				continue
			}
			if !paramValueEq(want, got) {
				addOnce("kernel %q param %q = %s, requested %s",
					kr.Name, name, formatValue(got), formatValue(want))
			}
		}
	}

	if out.Summary != nil && out.Summary.TotalTimeMS != nil {
		total := *out.Summary.TotalTimeMS
		if math.IsNaN(total) || math.IsInf(total, 0) || total < 0 {
			add("summary has invalid total_time_ms %v", total)
		} else if timesValid && len(out.Kernels) > 0 {
			if diff := math.Abs(total - kernelSum); diff > summaryTolerance(kernelSum, len(out.Kernels)) {
				add("summary total_time_ms %.6f != sum of kernel times %.6f",
					total, kernelSum)
			}
		}
	}

	return violations
}

// joinLimited joins up to max items with "; ", noting how many were elided.
func joinLimited(items []string, max int) string {
	if len(items) <= max {
		return strings.Join(items, "; ")
	}
	return fmt.Sprintf("%s; (+%d more)", strings.Join(items[:max], "; "), len(items)-max)
}
