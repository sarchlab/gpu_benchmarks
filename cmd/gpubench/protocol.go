package main

import (
	"encoding/json"
	"strings"
)

// KernelResult represents a parsed kernel event.
type KernelResult struct {
	Name    string  `json:"name"`
	TimeMS  float64 `json:"time_ms"`
	Ordinal int     `json:"-"`
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

// ErrorResult represents a parsed error event.
type ErrorResult struct {
	Message string `json:"message"`
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
			if t, ok := obj["time_ms"].(float64); ok {
				timeMS = t
			}
			kernels = append(kernels, KernelResult{
				Name:    name,
				TimeMS:  timeMS,
				Ordinal: ordinal,
			})
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
			errors = append(errors, ErrorResult{Message: msg})
		}
	}

	return kernels, summary, errors
}
