package main

import (
	"strings"
	"testing"
)

// validOutput is a protocol-compliant two-kernel run whose summary equals the
// sum of the kernel times and whose params echo the requested config.
const validOutput = `{"type":"info","benchmark":"demo"}
{"type":"kernel","name":"k1","iter":0,"time_ms":1.5,"host_time_ms":2.0,"params":{"size":256,"block_size":16}}
{"type":"kernel","name":"k2","iter":0,"time_ms":0.5,"host_time_ms":0.7,"params":{"size":256,"block_size":16}}
{"type":"summary","total_time_ms":2.0}
`

func demoExpected() map[string]interface{} {
	return map[string]interface{}{"size": float64(256), "block_size": float64(16)}
}

var demoKernels = []string{"k1", "k2"}

func validateStdout(stdout string) []string {
	return validateRun(parseOutput(stdout), demoKernels, demoExpected())
}

func requireViolation(t *testing.T, violations []string, substr string) {
	t.Helper()
	for _, v := range violations {
		if strings.Contains(v, substr) {
			return
		}
	}
	t.Fatalf("no violation containing %q in %v", substr, violations)
}

func TestValidateRunAcceptsCompliantOutput(t *testing.T) {
	if v := validateStdout(validOutput); len(v) != 0 {
		t.Fatalf("compliant output rejected: %v", v)
	}
}

func TestValidateRunRejections(t *testing.T) {
	cases := []struct {
		name    string
		stdout  string
		wantSub string
	}{
		{
			"no summary",
			`{"type":"kernel","name":"k1","time_ms":1.0,"params":{"size":256,"block_size":16}}`,
			"no summary event",
		},
		{
			"multiple summaries",
			validOutput + `{"type":"summary","total_time_ms":2.0}`,
			"2 summary events",
		},
		{
			"no kernels",
			`{"type":"summary","total_time_ms":0.0}`,
			"no kernel events",
		},
		{
			"summary missing total_time_ms",
			`{"type":"kernel","name":"k1","time_ms":1.0,"params":{"size":256,"block_size":16}}` + "\n" +
				`{"type":"summary"}`,
			"total_time_ms",
		},
		{
			"summary != kernel sum",
			`{"type":"kernel","name":"k1","time_ms":1.0,"params":{"size":256,"block_size":16}}` + "\n" +
				`{"type":"summary","total_time_ms":5.0}`,
			"sum of kernel times",
		},
		{
			"negative kernel time",
			`{"type":"kernel","name":"k1","time_ms":-1.0,"params":{"size":256,"block_size":16}}` + "\n" +
				`{"type":"summary","total_time_ms":-1.0}`,
			"invalid time_ms",
		},
		{
			"negative summary total",
			`{"type":"kernel","name":"k1","time_ms":1.0,"params":{"size":256,"block_size":16}}` + "\n" +
				`{"type":"summary","total_time_ms":-1.0}`,
			"invalid total_time_ms",
		},
		{
			"negative host time",
			`{"type":"kernel","name":"k1","time_ms":1.0,"host_time_ms":-2.0,"params":{"size":256,"block_size":16}}` + "\n" +
				`{"type":"summary","total_time_ms":1.0}`,
			"invalid host_time_ms",
		},
		{
			"missing time_ms",
			`{"type":"kernel","name":"k1","params":{"size":256,"block_size":16}}` + "\n" +
				`{"type":"summary","total_time_ms":0.0}`,
			"missing required time_ms",
		},
		{
			"undeclared kernel name",
			`{"type":"kernel","name":"rogue","time_ms":1.0,"params":{"size":256,"block_size":16}}` + "\n" +
				`{"type":"summary","total_time_ms":1.0}`,
			`kernel name "rogue" not in declared`,
		},
		{
			"missing params object",
			`{"type":"kernel","name":"k1","time_ms":1.0}` + "\n" +
				`{"type":"summary","total_time_ms":1.0}`,
			"missing required params object",
		},
		{
			"non-object params",
			`{"type":"kernel","name":"k1","time_ms":1.0,"params":[1,2]}` + "\n" +
				`{"type":"summary","total_time_ms":1.0}`,
			"params is not a JSON object",
		},
		{
			"missing expected param",
			`{"type":"kernel","name":"k1","time_ms":1.0,"params":{"size":256}}` + "\n" +
				`{"type":"summary","total_time_ms":1.0}`,
			`params missing "block_size"`,
		},
		{
			"mismatched param value",
			`{"type":"kernel","name":"k1","time_ms":1.0,"params":{"size":512,"block_size":16}}` + "\n" +
				`{"type":"summary","total_time_ms":1.0}`,
			`param "size" = 512, requested 256`,
		},
		{
			"fatal error with exit 0",
			validOutput + `{"type":"error","severity":"fatal","message":"driver crash"}`,
			"fatal error event reported despite exit code 0",
		},
		{
			"non-JSON line",
			validOutput + "Segmentation fault\n",
			"non-JSON stdout line",
		},
		{
			"unknown event type",
			validOutput + `{"type":"mystery"}`,
			`unknown event type "mystery"`,
		},
		{
			"kernel missing name",
			`{"type":"kernel","time_ms":1.0,"params":{"size":256,"block_size":16}}` + "\n" +
				`{"type":"summary","total_time_ms":1.0}`,
			"kernel event missing required name",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := validateStdout(tc.stdout)
			if len(v) == 0 {
				t.Fatalf("expected rejection, got none")
			}
			requireViolation(t, v, tc.wantSub)
		})
	}
}

func TestValidateRunWarningErrorAllowed(t *testing.T) {
	// A warning-severity error event does not invalidate an otherwise
	// complete run.
	stdout := validOutput + `{"type":"error","severity":"warning","message":"minor"}`
	if v := validateStdout(stdout); len(v) != 0 {
		t.Fatalf("warning event should not reject run: %v", v)
	}
}

func TestValidateRunPerIterationSum(t *testing.T) {
	// Per-iteration benchmarks emit one kernel event per timed iteration;
	// the summary is the sum over all of them.
	stdout := `{"type":"kernel","name":"k1","iter":0,"time_ms":1.0,"params":{"size":256,"block_size":16}}
{"type":"kernel","name":"k1","iter":1,"time_ms":1.25,"params":{"size":256,"block_size":16}}
{"type":"summary","total_time_ms":2.25}`
	if v := validateStdout(stdout); len(v) != 0 {
		t.Fatalf("per-iteration output rejected: %v", v)
	}
}

func TestValidateRunSummaryTolerance(t *testing.T) {
	// %.6f round-tripping wiggle must be tolerated.
	stdout := `{"type":"kernel","name":"k1","time_ms":1.000001,"params":{"size":256,"block_size":16}}
{"type":"summary","total_time_ms":1.000002}`
	if v := validateStdout(stdout); len(v) != 0 {
		t.Fatalf("within-tolerance summary rejected: %v", v)
	}
}

func TestValidateRunDedupesRepeatedViolations(t *testing.T) {
	// The same defect on every per-iteration event is reported once.
	var lines []string
	for i := 0; i < 10; i++ {
		lines = append(lines,
			`{"type":"kernel","name":"k1","iter":`+string(rune('0'+i))+
				`,"time_ms":1.0,"params":{"size":512,"block_size":16}}`)
	}
	lines = append(lines, `{"type":"summary","total_time_ms":10.0}`)
	v := validateStdout(strings.Join(lines, "\n"))
	count := 0
	for _, msg := range v {
		if strings.Contains(msg, `param "size"`) {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("param mismatch reported %d times, want 1: %v", count, v)
	}
}

func TestExpectedRunParams(t *testing.T) {
	exp := expectedRunParams("size", float64(256), map[string]interface{}{
		"block_size":        float64(16),
		"vector_add_warmup": "true", // measurement control: excluded
	})
	if len(exp) != 2 || exp["size"] != float64(256) || exp["block_size"] != float64(16) {
		t.Fatalf("expectedRunParams = %v", exp)
	}
}

func TestParamValueEq(t *testing.T) {
	cases := []struct {
		want, got interface{}
		eq        bool
	}{
		{float64(256), float64(256), true},
		{float64(256), float64(512), false},
		{float64(0.85), float64(0.85), true},
		{"read", "read", true},
		{"read", "write", false},
		// Numeric string on either side coerces to a numeric comparison.
		{"256", float64(256), true},
		{float64(1.5), "1.5", true},
		{float64(1.5), "1.6", false},
	}
	for _, tc := range cases {
		if got := paramValueEq(tc.want, tc.got); got != tc.eq {
			t.Fatalf("paramValueEq(%v, %v) = %v, want %v", tc.want, tc.got, got, tc.eq)
		}
	}
}

func TestJoinLimited(t *testing.T) {
	items := []string{"a", "b", "c", "d"}
	if got := joinLimited(items, 8); got != "a; b; c; d" {
		t.Fatalf("joinLimited under cap = %q", got)
	}
	if got := joinLimited(items, 2); got != "a; b; (+2 more)" {
		t.Fatalf("joinLimited over cap = %q", got)
	}
}
