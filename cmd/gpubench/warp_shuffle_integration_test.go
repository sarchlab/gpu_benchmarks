package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// Exercise the real GPU binary through the same parser and acceptance gate as
// gpubench run. Opt in on a CUDA/ROCm host; ordinary CPU tests remain GPU-free.
func TestWarpShuffleOutput(t *testing.T) {
	binary := os.Getenv("GPUBENCH_WARP_SHUFFLE_BINARY")
	if binary == "" {
		t.Skip("set GPUBENCH_WARP_SHUFFLE_BINARY to a built warp_shuffle binary")
	}
	binary, err := filepath.Abs(binary)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(filepath.Dir(binary), "params.json"))
	if err != nil {
		t.Fatal(err)
	}
	var params BenchmarkParams
	if err := json.Unmarshal(data, &params); err != nil {
		t.Fatal(err)
	}
	if params.Benchmark != "warp_shuffle" || len(params.ScalingParam.Values) == 0 {
		t.Fatal("expected a nonempty warp_shuffle manifest beside the binary")
	}
	sizes := params.ScalingParam.Values
	if os.Getenv("GPUBENCH_GPU_SWEEP") != "1" {
		sizes = []interface{}{sizes[0], sizes[len(sizes)-1]}
	}
	configs := generateConfigs(&params, true, sizes,
		map[string]string{"vector_add_warmup": "false"})
	for _, config := range configs {
		variant := config.NSDict["shuffle_variant"].(string)
		name := fmt.Sprintf("%v/%v/%s", config.ScalingValue,
			config.NSDict["threads_per_block"], variant)
		t.Run(name, func(t *testing.T) {
			exitCode, stdout, stderr, timedOut := runBenchmarkProc([]string{binary},
				buildEnv(params.ScalingParam.Name, config.ScalingValue, config.NSDict),
				30, false, filepath.Dir(binary))
			if exitCode != 0 || timedOut {
				t.Fatalf("exit=%d timeout=%v: %s\n%s", exitCode, timedOut, stdout, stderr)
			}
			wantKernels := params.Kernels
			if variant != "all" {
				wantKernels = []string{variant + "_kernel"}
			}
			out := parseOutput(stdout)
			expected := expectedRunParams(params.ScalingParam.Name,
				config.ScalingValue, config.NSDict)
			if violations := validateRun(out, wantKernels, expected); len(violations) != 0 {
				t.Fatalf("output rejected: %v\n%s", violations, stdout)
			}
			if len(out.Kernels) != len(wantKernels) {
				t.Fatalf("got %d kernel records, want %d", len(out.Kernels), len(wantKernels))
			}
		})
	}
	t.Logf("checked %d GPU configurations against the production output validator", len(configs))
}
