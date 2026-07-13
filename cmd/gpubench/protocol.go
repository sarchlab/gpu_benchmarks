package main

import (
	"encoding/json"
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
	// HasTime records whether the required "time_ms" field was present, so the
	// harness can distinguish a real 0 ms from a missing field.
	HasTime bool `json:"-"`
}

// SummaryResult represents a parsed summary event.
type SummaryResult struct {
	TotalTimeMS *float64        `json:"total_time_ms"`
	Metrics     []MetricResult  `json:"metrics"`
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

// parseOutput parses JSON-lines stdout. Returns (kernels, summary, errors).
func parseOutput(stdoutData string) ([]KernelResult, *SummaryResult, []ErrorResult) {
	var kernels []KernelResult
	var summary *SummaryResult
	var errors []ErrorResult
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
			// lost/truncated record. Warn rather than silently drop it.
			preview := line
			if len(preview) > 120 {
				preview = preview[:120] + "..."
			}
			eprintf("warning: skipping non-JSON stdout line: %s\n", preview)
			continue
		}

		typeStr, _ := obj["type"].(string)
		switch typeStr {
		case "kernel":
			name, _ := obj["name"].(string)
			if name == "" {
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
				eprintf("warning: kernel event %q missing required time_ms; recording 0.0\n", name)
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
				if b, err := json.Marshal(p); err == nil {
					kr.Params = string(b)
				}
			}
			kernels = append(kernels, kr)
			ordinal++

		case "summary":
			s := &SummaryResult{}
			if t, ok := obj["total_time_ms"].(float64); ok {
				s.TotalTimeMS = &t
			}
			if metricsRaw, ok := obj["metrics"].([]interface{}); ok {
				for _, m := range metricsRaw {
					if mObj, ok := m.(map[string]interface{}); ok {
						name, _ := mObj["name"].(string)
						if name == "" {
							name = "unknown"
						}
						value := 0.0
						if v, ok := mObj["value"].(float64); ok {
							value = v
						}
						s.Metrics = append(s.Metrics, MetricResult{
							Name:  name,
							Value: value,
						})
					}
				}
			}
			summary = s

		case "error":
			msg, _ := obj["message"].(string)
			sev, _ := obj["severity"].(string)
			errors = append(errors, ErrorResult{Message: msg, Severity: sev})
		}
	}

	return kernels, summary, errors
}
