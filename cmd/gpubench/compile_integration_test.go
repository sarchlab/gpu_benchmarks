package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Use real make targets: changing only command-line/environment build settings
// must rebuild Tier 1/2, despite a newer output and unchanged source timestamps.
func TestCompileRefreshesTier12(t *testing.T) {
	root := t.TempDir()
	binary := filepath.Join(root, "gpubench")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	build := exec.CommandContext(ctx, "go", "build", "-buildvcs=false", "-o", binary, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build CLI: %v\n%s", err, out)
	}

	fixture := func(tier int, name, suite, language string) string {
		dir := filepath.Join(root, fmt.Sprintf("tier%d", tier), name)
		manifest, err := json.Marshal(BenchmarkParams{
			Benchmark: name, Tier: tier, Suite: suite, Language: language,
		})
		if err != nil {
			t.Fatal(err)
		}
		writeTestFile(t, filepath.Join(dir, "params.json"), string(manifest))
		writeTestFile(t, filepath.Join(dir, "source"), "unchanged source\n")
		writeTestFile(t, filepath.Join(dir, "Makefile"), `artifact: source
	@test "$(TEST_COMPILE_FAIL)" != "1"
	@printf '%s\n' '$(PLATFORM) $(CUDA_ARCH) $(TEST_COMPILE_FLAGS)' > $@
	@printf '%s\n' '$(MAKEFLAGS)' >> builds
`)
		return dir
	}
	one := fixture(1, "one", "selected", "cpp")
	two := fixture(2, "two", "selected", "cpp")
	three := fixture(3, "three", "selected", "cpp")
	python := fixture(2, "python", "selected", "python")
	other := fixture(2, "other", "excluded", "cpp")

	run := func(arch, flags string, fail bool, want int, args ...string) {
		t.Helper()
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, binary, append([]string{"compile"}, args...)...)
		cmd.Dir = root
		failValue := "0"
		if fail {
			failValue = "1"
		}
		cmd.Env = append(os.Environ(), "MAKEFLAGS=", "MFLAGS=", "MAKEOVERRIDES=",
			"CUDA_ARCH="+arch, "TEST_COMPILE_FLAGS="+flags, "TEST_COMPILE_FAIL="+failValue)
		out, err := cmd.CombinedOutput()
		if cmd.ProcessState == nil || cmd.ProcessState.ExitCode() != want || ctx.Err() != nil {
			t.Fatalf("compile %v: want exit %d, got %v\n%s", args, want, err, out)
		}
	}
	check := func(dir, want string, builds int) string {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(dir, "artifact"))
		if err != nil || string(data) != want+"\n" {
			t.Fatalf("%s artifact: %q, %v; want %q", dir, data, err, want)
		}
		log, err := os.ReadFile(filepath.Join(dir, "builds"))
		if err != nil || strings.Count(string(log), "\n") != builds {
			t.Fatalf("%s build count: %q, %v; want %d", dir, log, err, builds)
		}
		return string(log)
	}

	run("sm_80", "first", false, ExitOK, "--suite", "selected", "--platform", "cuda")
	for _, dir := range []string{one, two, three} {
		check(dir, "cuda sm_80 first", 1)
	}
	for _, item := range []struct{ tier, dir string }{{"1", one}, {"2", two}} {
		t.Logf("Checking tier %s rebuilds", item.tier)
		args := []string{"--tier", item.tier, "--suite", "selected", "--platform", "cuda", "--jobs", "2"}
		run("sm_90", "first", false, ExitOK, args...)
		log := check(item.dir, "cuda sm_90 first", 2)
		if !strings.Contains(log, "-j") {
			t.Fatalf("parallel jobs not passed to make: %q", log)
		}
		// Compiler settings and same-target repeats must also refresh.
		run("sm_90", "second", false, ExitOK, args...)
		check(item.dir, "cuda sm_90 second", 3)
		run("sm_90", "second", false, ExitOK, args...)
		check(item.dir, "cuda sm_90 second", 4)
		args[5] = "rocm"
		run("sm_90", "second", false, ExitOK, args...)
		check(item.dir, "rocm sm_90 second", 5)
		// A failed requested build must not succeed by reusing the old artifact.
		run("sm_90", "second", true, ExitSomeFailed, args...)
		check(item.dir, "rocm sm_90 second", 5)
	}
	// Benchmark selection and automatic platform detection still work.
	run("sm_90", "auto", false, ExitOK, "--benchmark", "one")
	check(one, detectPlatform()+" sm_90 auto", 6)
	check(two, "rocm sm_90 second", 5)
	// Preserve incremental behavior outside Tier 1/2, even when settings change.
	run("sm_90", "second", false, ExitOK, "--tier", "3", "--platform", "rocm")
	check(three, "cuda sm_80 first", 1)
	run("sm_90", "second", true, ExitOK, "--benchmark", "python", "--platform", "cuda")
	run("sm_90", "second", false, ExitNoBench, "--benchmark", "absent", "--platform", "cuda")
	for _, dir := range []string{python, other} {
		if _, err := os.Stat(filepath.Join(dir, "artifact")); !os.IsNotExist(err) {
			t.Fatalf("unexpected artifact in skipped/filtered fixture %s: %v", dir, err)
		}
	}
}
