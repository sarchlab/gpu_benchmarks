package main

import (
	"encoding/json"
	"path/filepath"
	"testing"
)

// writeVariantTree creates tier1/single (one source tree), tier2/both
// (hip/ and cuda/) and tier2/hiponly (hip/ only) under a temp root.
func writeVariantTree(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	manifest := func(bench string, tier float64) string {
		p := baseValidParams(bench)
		p["tier"] = tier
		data, err := json.Marshal(p)
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	writeTestFile(t, filepath.Join(root, "tier1", "single", "params.json"), manifest("single", 1))
	writeTestFile(t, filepath.Join(root, "tier2", "both", "hip", "params.json"), manifest("both", 2))
	writeTestFile(t, filepath.Join(root, "tier2", "both", "cuda", "params.json"), manifest("both", 2))
	writeTestFile(t, filepath.Join(root, "tier2", "hiponly", "hip", "params.json"), manifest("hiponly", 2))
	return root
}

func TestDiscoverBenchmarksSelectsPlatformVariant(t *testing.T) {
	root := writeVariantTree(t)
	for _, tc := range []struct {
		platform string
		want     map[string]string // benchmark -> directory relative to root
	}{
		{"cuda", map[string]string{
			"single":  "tier1/single",
			"both":    "tier2/both/cuda",
			"hiponly": "tier2/hiponly/hip",
		}},
		{"rocm", map[string]string{
			"single":  "tier1/single",
			"both":    "tier2/both/hip",
			"hiponly": "tier2/hiponly/hip",
		}},
	} {
		t.Run(tc.platform, func(t *testing.T) {
			got := discoverBenchmarks(root, nil, "", "", tc.platform)
			if len(got) != len(tc.want) {
				t.Fatalf("discovered %d benchmarks, want %d", len(got), len(tc.want))
			}
			for _, b := range got {
				rel, _ := filepath.Rel(root, b.Dir)
				if want := tc.want[b.Benchmark]; filepath.ToSlash(rel) != want {
					t.Fatalf("%s: dir %s, want %s", b.Benchmark, rel, want)
				}
			}
		})
	}
}

func TestDiscoverBenchmarksSkipsCudaOnlyOnRocm(t *testing.T) {
	root := t.TempDir()
	data, err := json.Marshal(baseValidParams("cudaonly"))
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, filepath.Join(root, "tier1", "cudaonly", "cuda", "params.json"), string(data))
	if got := discoverBenchmarks(root, nil, "", "", "rocm"); len(got) != 0 {
		t.Fatalf("rocm discovered a cuda-only benchmark: %d", len(got))
	}
	if got := discoverBenchmarks(root, nil, "", "", "cuda"); len(got) != 1 {
		t.Fatalf("cuda discovered %d benchmarks, want 1", len(got))
	}
}

func TestDiscoverBenchmarksHonorsPlatformsList(t *testing.T) {
	root := t.TempDir()
	p := baseValidParams("rocmonly")
	p["platforms"] = []interface{}{"rocm"}
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	// hip/ is the tree cuda falls back to, so only the list keeps cuda away.
	writeTestFile(t, filepath.Join(root, "tier1", "rocmonly", "hip", "params.json"), string(data))
	if got := discoverBenchmarks(root, nil, "", "", "cuda"); len(got) != 0 {
		t.Fatalf("cuda discovered a rocm-only benchmark: %d", len(got))
	}
	if got := discoverBenchmarks(root, nil, "", "rocmonly", "rocm"); len(got) != 1 {
		t.Fatalf("rocm discovered %d benchmarks, want 1", len(got))
	}
}

func TestValidateParamsChecksPlatformsList(t *testing.T) {
	for _, tc := range []struct {
		name      string
		platforms interface{}
		wantErr   string
	}{
		{"one platform", []interface{}{"rocm"}, ""},
		{"both platforms", []interface{}{"cuda", "rocm"}, ""},
		{"unknown platform", []interface{}{"hip"}, "platforms[0] must be cuda or rocm"},
		{"non-string entry", []interface{}{float64(1)}, "platforms[0] must be cuda or rocm"},
		{"empty list", []interface{}{}, "platforms must be a nonempty array"},
		{"not a list", "rocm", "platforms must be a nonempty array"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := baseValidParams("mybench")
			p["platforms"] = tc.platforms
			errs, _ := validateParamsJSON(t, "mybench", p)
			if tc.wantErr == "" && len(errs) != 0 {
				t.Fatalf("unexpected errors: %v", errs)
			}
			if tc.wantErr != "" && !hasErrContaining(errs, tc.wantErr) {
				t.Fatalf("expected %q, got: %v", tc.wantErr, errs)
			}
		})
	}
}

func TestValidateManifestFilesChecksBothVariants(t *testing.T) {
	root := writeVariantTree(t)
	if got := validateManifestFiles(root, nil, "", ""); got != ExitOK {
		t.Fatalf("valid variant tree: exit=%d", got)
	}
	two := 2
	if got := validateManifestFiles(root, &two, "", "both"); got != ExitOK {
		t.Fatalf("tier/benchmark filter on variant: exit=%d", got)
	}

	// A cuda/ manifest that drifts from hip/ is an error.
	p := baseValidParams("both")
	p["tier"] = float64(2)
	p["description"] = "drifted"
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, filepath.Join(root, "tier2", "both", "cuda", "params.json"), string(data))
	if got := validateManifestFiles(root, nil, "", ""); got != ExitSomeFailed {
		t.Fatalf("drifted cuda manifest: exit=%d, want failure", got)
	}
}
