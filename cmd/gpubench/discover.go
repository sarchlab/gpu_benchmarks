package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
)

// ScalingParam represents the scaling parameter in params.json.
type ScalingParam struct {
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Type        string        `json:"type"`
	Values      []interface{} `json:"values"`
}

// NonScalingParam represents a non-scaling parameter in params.json.
type NonScalingParam struct {
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Type        string        `json:"type"`
	Default     interface{}   `json:"default"`
	Values      []interface{} `json:"values"`
}

// BenchmarkParams represents a parsed params.json.
type BenchmarkParams struct {
	Benchmark       string            `json:"benchmark"`
	Tier            int               `json:"tier"`
	Suite           string            `json:"suite"`
	Description     string            `json:"description"`
	ScalingParam    ScalingParam      `json:"scaling_param"`
	NonScalingParams []NonScalingParam `json:"non_scaling_params"`
	Kernels         []string          `json:"kernels"`
	Language        string            `json:"language,omitempty"`

	// Internal fields (not from JSON)
	Dir        string `json:"-"`
	ParamsPath string `json:"-"`
}

// getLanguage returns the benchmark language.
func getLanguage(params *BenchmarkParams) string {
	if params.Language != "" {
		return params.Language
	}
	if params.Tier >= 4 {
		return "python"
	}
	return "cpp"
}

// findRepoRoot walks up from executable location to find directory containing tier1/.
func findRepoRoot() string {
	// Start from executable location
	exe, err := os.Executable()
	if err == nil {
		d := filepath.Dir(exe)
		for i := 0; i < 10; i++ {
			if info, err := os.Stat(filepath.Join(d, "tier1")); err == nil && info.IsDir() {
				return d
			}
			parent := filepath.Dir(d)
			if parent == d {
				break
			}
			d = parent
		}
	}

	// Fall back to cwd
	cwd, err := os.Getwd()
	if err == nil {
		if info, err := os.Stat(filepath.Join(cwd, "tier1")); err == nil && info.IsDir() {
			return cwd
		}
	}

	eprint("Error: cannot locate repo root (expected directory containing tier1/)")
	os.Exit(ExitUsage)
	return ""
}

// discoverBenchmarks finds all benchmarks matching the given filters.
func discoverBenchmarks(repoRoot string, tier *int, suite, benchmark string) []*BenchmarkParams {
	pattern := filepath.Join(repoRoot, "tier*", "*", "params.json")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil
	}
	sort.Strings(matches)

	var results []*BenchmarkParams
	for _, pp := range matches {
		data, err := os.ReadFile(pp)
		if err != nil {
			continue
		}

		var params BenchmarkParams
		if err := json.Unmarshal(data, &params); err != nil {
			continue
		}

		if tier != nil && params.Tier != *tier {
			continue
		}
		if suite != "" && params.Suite != suite {
			continue
		}
		if benchmark != "" && params.Benchmark != benchmark {
			continue
		}

		params.Dir = filepath.Dir(pp)
		params.ParamsPath = pp
		results = append(results, &params)
	}
	return results
}

// parseTier parses a tier string to int pointer.
func parseTier(tierStr string) *int {
	if tierStr == "" {
		return nil
	}
	t, err := strconv.Atoi(tierStr)
	if err != nil {
		return nil
	}
	return &t
}

// configPair holds a scaling value and non-scaling parameter dict.
type configPair struct {
	ScalingValue interface{}
	NSDict       map[string]interface{}
}

// generateConfigs generates (scaling_value, non_scaling_dict) tuples.
func generateConfigs(params *BenchmarkParams, sweepNS bool, scalingOverride []interface{}) []configPair {
	svList := params.ScalingParam.Values
	if scalingOverride != nil {
		svList = scalingOverride
	}

	nspList := params.NonScalingParams
	if len(nspList) == 0 {
		var configs []configPair
		for _, sv := range svList {
			configs = append(configs, configPair{
				ScalingValue: sv,
				NSDict:       map[string]interface{}{},
			})
		}
		return configs
	}

	if sweepNS {
		// Cartesian product of non-scaling param values
		names := make([]string, len(nspList))
		valLists := make([][]interface{}, len(nspList))
		for i, p := range nspList {
			names[i] = p.Name
			valLists[i] = p.Values
		}

		var configs []configPair
		for _, sv := range svList {
			combos := cartesianProduct(valLists)
			for _, combo := range combos {
				nsDict := make(map[string]interface{})
				for i, name := range names {
					nsDict[name] = combo[i]
				}
				configs = append(configs, configPair{
					ScalingValue: sv,
					NSDict:       nsDict,
				})
			}
		}
		return configs
	}

	// Default mode: use default values for non-scaling params
	defaults := make(map[string]interface{})
	for _, p := range nspList {
		defaults[p.Name] = p.Default
	}

	var configs []configPair
	for _, sv := range svList {
		nsDict := make(map[string]interface{})
		for k, v := range defaults {
			nsDict[k] = v
		}
		configs = append(configs, configPair{
			ScalingValue: sv,
			NSDict:       nsDict,
		})
	}
	return configs
}

// cartesianProduct computes the cartesian product of multiple value lists.
func cartesianProduct(lists [][]interface{}) [][]interface{} {
	if len(lists) == 0 {
		return [][]interface{}{{}}
	}

	result := [][]interface{}{{}}
	for _, list := range lists {
		var newResult [][]interface{}
		for _, existing := range result {
			for _, val := range list {
				combo := make([]interface{}, len(existing)+1)
				copy(combo, existing)
				combo[len(existing)] = val
				newResult = append(newResult, combo)
			}
		}
		result = newResult
	}
	return result
}

// smallestScalingConfigs keeps only configs whose scaling value is among the n
// smallest distinct values (numeric order when all values parse as numbers,
// otherwise the order they were listed in params.json). Used by trace
// collection to trace a few small sizes per benchmark and let `ptrace
// scale`/`gen` extrapolate the rest. Pick sizes in the same compiled regime as
// the targets — gate with `ptrace regime` (a remainder-loop tiny size cannot
// generate an unrolled large one).
func smallestScalingConfigs(configs []configPair, n int) []configPair {
	var distinct []interface{}
	seen := map[string]bool{}
	for _, c := range configs {
		k := formatValue(c.ScalingValue)
		if !seen[k] {
			seen[k] = true
			distinct = append(distinct, c.ScalingValue)
		}
	}
	allNumeric := true
	nums := make([]float64, len(distinct))
	for i, v := range distinct {
		f, err := strconv.ParseFloat(formatValue(v), 64)
		if err != nil {
			allNumeric = false
			break
		}
		nums[i] = f
	}
	if allNumeric {
		idx := make([]int, len(distinct))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return nums[idx[a]] < nums[idx[b]] })
		sorted := make([]interface{}, len(distinct))
		for i, j := range idx {
			sorted[i] = distinct[j]
		}
		distinct = sorted
	}
	if n > len(distinct) {
		n = len(distinct)
	}
	keep := map[string]bool{}
	for _, v := range distinct[:n] {
		keep[formatValue(v)] = true
	}
	var out []configPair
	for _, c := range configs {
		if keep[formatValue(c.ScalingValue)] {
			out = append(out, c)
		}
	}
	return out
}

// formatValue formats an interface{} value for display.
func formatValue(v interface{}) string {
	switch val := v.(type) {
	case float64:
		if val == float64(int64(val)) {
			return strconv.FormatInt(int64(val), 10)
		}
		return strconv.FormatFloat(val, 'f', -1, 64)
	case int:
		return strconv.Itoa(val)
	case int64:
		return strconv.FormatInt(val, 10)
	case string:
		return val
	default:
		return fmt.Sprintf("%v", v)
	}
}

// sortValues sorts interface values: numbers first, then strings.
func sortValues(vals []interface{}) {
	sort.Slice(vals, func(i, j int) bool {
		vi := toFloat(vals[i])
		vj := toFloat(vals[j])
		viIsNum := vi != nil
		vjIsNum := vj != nil
		if viIsNum && vjIsNum {
			return *vi < *vj
		}
		if viIsNum && !vjIsNum {
			return true
		}
		if !viIsNum && vjIsNum {
			return false
		}
		return fmt.Sprintf("%v", vals[i]) < fmt.Sprintf("%v", vals[j])
	})
}

func toFloat(v interface{}) *float64 {
	switch val := v.(type) {
	case float64:
		return &val
	case int:
		f := float64(val)
		return &f
	case int64:
		f := float64(val)
		return &f
	default:
		return nil
	}
}
