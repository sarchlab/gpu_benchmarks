package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ── Command: list ──────────────────────────────────────────────────────────

func cmdList(tierStr, suite string, jsonOut bool) int {
	repoRoot := findRepoRoot()
	tier := parseTier(tierStr)
	benchmarks := discoverBenchmarks(repoRoot, tier, suite, "")
	if len(benchmarks) == 0 {
		eprint("No benchmarks found.")
		return ExitNoBench
	}

	if jsonOut {
		type listEntry struct {
			Tier         int    `json:"tier"`
			Suite        string `json:"suite"`
			Benchmark    string `json:"benchmark"`
			ScalingParam string `json:"scaling_param"`
			Kernels      int    `json:"kernels"`
			Language     string `json:"language"`
			Description  string `json:"description"`
		}
		var out []listEntry
		for _, b := range benchmarks {
			out = append(out, listEntry{
				Tier:         b.Tier,
				Suite:        b.Suite,
				Benchmark:    b.Benchmark,
				ScalingParam: b.ScalingParam.Name,
				Kernels:      len(b.Kernels),
				Language:     getLanguage(b),
				Description:  b.Description,
			})
		}
		data, _ := json.MarshalIndent(out, "", "  ")
		fmt.Println(string(data))
		return ExitOK
	}

	fmtStr := "%-5v %-18v %-28v %-20v %-8v %v"
	fmt.Println(fmt.Sprintf(fmtStr, "Tier", "Suite", "Benchmark", "Scaling Param", "Kernels", "Lang"))
	fmt.Println(fmt.Sprintf(fmtStr, "----", "-----", "---------", "-------------", "-------", "----"))

	tc := make(map[int]int)
	for _, b := range benchmarks {
		tc[b.Tier]++
		fmt.Println(fmt.Sprintf(fmtStr,
			b.Tier, b.Suite, b.Benchmark,
			b.ScalingParam.Name, len(b.Kernels), getLanguage(b)))
	}

	tiers := make([]int, 0, len(tc))
	for t := range tc {
		tiers = append(tiers, t)
	}
	sort.Ints(tiers)

	var parts []string
	total := 0
	for _, t := range tiers {
		parts = append(parts, fmt.Sprintf("%d tier%d", tc[t], t))
		total += tc[t]
	}
	fmt.Printf("\nTotal: %d benchmarks (%s)\n", total, strings.Join(parts, ", "))
	return ExitOK
}

// ── Command: compile ───────────────────────────────────────────────────────

func cmdCompile(tierStr, suite, benchmark, platform string, jobs int, verbose bool) int {
	repoRoot := findRepoRoot()
	tier := parseTier(tierStr)
	plat := platform
	if plat == "" {
		plat = detectPlatform()
	}

	benchmarks := discoverBenchmarks(repoRoot, tier, suite, benchmark)
	if len(benchmarks) == 0 {
		eprint("No benchmarks found matching filters.")
		return ExitNoBench
	}

	// Filter to C++ only
	var cpp []*BenchmarkParams
	for _, b := range benchmarks {
		if getLanguage(b) != "python" {
			cpp = append(cpp, b)
		}
	}
	skipped := len(benchmarks) - len(cpp)

	if len(cpp) == 0 {
		eprintf("No C++ benchmarks to compile (skipped %d Python).\n", skipped)
		return ExitOK
	}

	eprintf("=== Compiling %d benchmarks (platform: %s) ===\n", len(cpp), plat)
	if skipped > 0 {
		eprintf("  (skipping %d Python benchmarks)\n", skipped)
	}

	passed, failed := 0, 0
	var failures []string

	// Open log file for compilation errors
	logFile, logErr := os.OpenFile("compile_errors.log",
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if logErr != nil {
		eprintf("Warning: could not open compile_errors.log: %v\n", logErr)
	} else {
		defer logFile.Close()
	}

	for i, b := range cpp {
		eprintf("[%3d/%d] %s ... ", i+1, len(cpp), b.Benchmark)

		args := []string{"-C", b.Dir, fmt.Sprintf("PLATFORM=%s", plat)}
		if jobs > 0 {
			args = append(args, fmt.Sprintf("-j%d", jobs))
		}

		ctx, cancel := context.WithTimeout(context.Background(), 600*time.Second)
		cmd := exec.CommandContext(ctx, "make", args...)
		cmd.Stdout = nil
		var stderrBuf strings.Builder
		cmd.Stderr = &stderrBuf

		err := cmd.Run()
		cancel()

		if ctx.Err() == context.DeadlineExceeded {
			eprint("✗ (timeout)")
			failed++
			failures = append(failures, b.Benchmark)
			if logFile != nil {
				fmt.Fprintf(logFile, "=== %s (timeout) ===\n", b.Benchmark)
			}
		} else if err != nil {
			eprint("✗")
			failed++
			failures = append(failures, b.Benchmark)
			errOutput := stderrBuf.String()
			if errOutput != "" {
				eprint(errOutput)
			}
			if logFile != nil {
				fmt.Fprintf(logFile, "=== %s ===\n%s\n", b.Benchmark, errOutput)
			}
		} else {
			eprint("✓")
			passed++
		}
	}

	eprint("\n=== Compile Summary ===")
	eprintf("  Passed: %d  Failed: %d\n", passed, failed)
	if len(failures) > 0 {
		eprintf("  Failed: %s\n", strings.Join(failures, ", "))
	}

	if failed > 0 {
		return ExitSomeFailed
	}
	return ExitOK
}

// ── Command: info ──────────────────────────────────────────────────────────

func cmdInfo(jsonOut bool) int {
	plat := detectPlatform()
	info := collectPlatformInfo(plat)

	if jsonOut {
		data, _ := json.MarshalIndent(info, "", "  ")
		fmt.Println(string(data))
		return ExitOK
	}

	fmt.Println("Platform Information")
	fmt.Printf("  %-14s %s\n", "Platform:", info.PlatformType)
	fmt.Printf("  %-14s %s\n", "GPU:", info.GPUModel)
	if info.GPUVramGB != nil {
		fmt.Printf("  %-14s %v GiB\n", "GPU VRAM:", *info.GPUVramGB)
	}
	if info.DriverVersion != nil {
		fmt.Printf("  %-14s %s\n", "Driver:", *info.DriverVersion)
	}
	if info.RuntimeVersion != nil {
		fmt.Printf("  %-14s %s\n", "Runtime:", *info.RuntimeVersion)
	}
	if info.Compiler != nil {
		fmt.Printf("  %-14s %s\n", "Compiler:", *info.Compiler)
	}
	fmt.Printf("  %-14s %s\n", "OS:", info.OS)
	if info.CPUModel != nil {
		cores := ""
		if info.CPUCores != nil {
			cores = fmt.Sprintf(" (%d cores)", *info.CPUCores)
		}
		fmt.Printf("  %-14s %s%s\n", "CPU:", *info.CPUModel, cores)
	}
	if info.RAMGB != nil {
		fmt.Printf("  %-14s %v GiB\n", "RAM:", *info.RAMGB)
	}
	fmt.Printf("  %-14s %s\n", "Hostname:", info.Hostname)
	fmt.Printf("  %-14s %s\n", "Go:", info.PythonVersion)
	return ExitOK
}

// ── Command: validate ──────────────────────────────────────────────────────

var identRE = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)

func validateParams(params map[string]interface{}, paramsPath string) []string {
	var errs []string
	bdir := filepath.Dir(paramsPath)
	dirName := filepath.Base(bdir)
	tierDir := filepath.Base(filepath.Dir(bdir))

	required := []string{"benchmark", "tier", "suite", "description",
		"scaling_param", "non_scaling_params", "kernels"}
	for _, f := range required {
		if _, ok := params[f]; !ok {
			errs = append(errs, fmt.Sprintf("Missing required field: '%s'", f))
		}
	}
	if len(errs) > 0 {
		return errs
	}

	// benchmark vs directory name
	benchName, _ := params["benchmark"].(string)
	if benchName != dirName {
		errs = append(errs, fmt.Sprintf("benchmark '%s' != directory '%s'", benchName, dirName))
	}

	// tier vs parent dir
	tierNum, _ := params["tier"].(float64) // JSON numbers are float64
	tierDirStr := strings.TrimPrefix(tierDir, "tier")
	if tn, err := strconv.Atoi(tierDirStr); err == nil {
		if int(tierNum) != tn {
			errs = append(errs, fmt.Sprintf("tier %d != parent dir '%s'", int(tierNum), tierDir))
		}
	} else {
		errs = append(errs, fmt.Sprintf("Parent dir '%s' not a tier pattern", tierDir))
	}

	// scaling_param checks
	sp, _ := params["scaling_param"].(map[string]interface{})
	if sp != nil {
		for _, f := range []string{"name", "description", "type", "values"} {
			if _, ok := sp[f]; !ok {
				errs = append(errs, fmt.Sprintf("scaling_param missing '%s'", f))
			}
		}
		if name, ok := sp["name"].(string); ok {
			if !identRE.MatchString(name) {
				errs = append(errs, fmt.Sprintf("scaling_param name '%s' not a valid identifier", name))
			}
		}
		if values, ok := sp["values"].([]interface{}); ok {
			if len(values) < 10 {
				errs = append(errs, fmt.Sprintf("scaling_param needs >= 10 values, has %d", len(values)))
			}
		}
	}

	// Collect all param names for duplicate check
	var allNames []string
	if sp != nil {
		if name, ok := sp["name"].(string); ok {
			allNames = append(allNames, name)
		}
	}

	// non_scaling_params checks
	nspList, _ := params["non_scaling_params"].([]interface{})
	for idx, nspRaw := range nspList {
		pfx := fmt.Sprintf("non_scaling_params[%d]", idx)
		nsp, ok := nspRaw.(map[string]interface{})
		if !ok {
			continue
		}
		for _, f := range []string{"name", "description", "type", "default", "values"} {
			if _, ok := nsp[f]; !ok {
				errs = append(errs, fmt.Sprintf("%s missing '%s'", pfx, f))
			}
		}
		if name, ok := nsp["name"].(string); ok {
			if !identRE.MatchString(name) {
				errs = append(errs, fmt.Sprintf("%s name '%s' not a valid identifier", pfx, name))
			}
			allNames = append(allNames, name)
		}
		// Check default in values
		if defVal, hasDef := nsp["default"]; hasDef {
			if values, hasVals := nsp["values"].([]interface{}); hasVals {
				found := false
				defJSON, _ := json.Marshal(defVal)
				for _, v := range values {
					vJSON, _ := json.Marshal(v)
					if string(defJSON) == string(vJSON) {
						found = true
						break
					}
				}
				if !found {
					defStr, _ := json.Marshal(defVal)
					errs = append(errs, fmt.Sprintf("%s default %s not in values", pfx, string(defStr)))
				}
			}
		}
	}

	// Duplicate name check
	seen := make(map[string]bool)
	for _, n := range allNames {
		if seen[n] {
			errs = append(errs, fmt.Sprintf("Duplicate parameter name: '%s'", n))
		}
		seen[n] = true
	}

	return errs
}

func cmdValidate(tierStr, suite, benchmark string) int {
	repoRoot := findRepoRoot()
	tier := parseTier(tierStr)
	benchmarks := discoverBenchmarks(repoRoot, tier, suite, benchmark)
	if len(benchmarks) == 0 {
		eprint("No benchmarks found matching filters.")
		return ExitNoBench
	}

	totalErrors, valid := 0, 0
	for _, b := range benchmarks {
		pp := b.ParamsPath
		rel, _ := filepath.Rel(repoRoot, pp)
		if rel == "" {
			rel = pp
		}

		data, err := os.ReadFile(pp)
		if err != nil {
			fmt.Printf("✗ %s — Invalid JSON: %v\n", rel, err)
			totalErrors++
			continue
		}

		var params map[string]interface{}
		if err := json.Unmarshal(data, &params); err != nil {
			fmt.Printf("✗ %s — Invalid JSON: %v\n", rel, err)
			totalErrors++
			continue
		}

		es := validateParams(params, pp)
		if len(es) > 0 {
			fmt.Printf("✗ %s\n", rel)
			for _, e := range es {
				fmt.Printf("    - %s\n", e)
			}
			totalErrors += len(es)
		} else {
			fmt.Printf("✓ %s\n", rel)
			valid++
		}
	}

	fmt.Printf("\n=== Validation Summary ===\n")
	fmt.Printf("  Valid:  %d/%d\n", valid, len(benchmarks))
	fmt.Printf("  Errors: %d\n", totalErrors)

	if totalErrors > 0 {
		return ExitSomeFailed
	}
	return ExitOK
}

// ── Benchmark execution helpers ────────────────────────────────────────────

func buildEnv(scalingName string, scalingValue interface{}, nsDict map[string]interface{}) []string {
	env := os.Environ()
	env = append(env, fmt.Sprintf("BENCH_PARAM_%s=%s", scalingName, formatValue(scalingValue)))
	for k, v := range nsDict {
		env = append(env, fmt.Sprintf("BENCH_PARAM_%s=%s", k, formatValue(v)))
	}
	return env
}

func getBenchCommand(params *BenchmarkParams, plat string) []string {
	lang := getLanguage(params)
	if lang == "python" {
		if plat == "metal" {
			mps := filepath.Join(params.Dir, "bench_mps.py")
			if _, err := os.Stat(mps); err == nil {
				return []string{"python3", mps}
			}
		}
		return []string{"python3", filepath.Join(params.Dir, "bench.py")}
	}
	return []string{filepath.Join(params.Dir, params.Benchmark)}
}

func runBenchmarkProc(cmd []string, env []string, timeoutSecs int, verbose bool) (exitCode int, stdout, stderr string, timedOut bool) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeoutSecs)*time.Second)
	defer cancel()

	proc := exec.CommandContext(ctx, cmd[0], cmd[1:]...)
	proc.Env = env

	var stdoutBuf, stderrBuf strings.Builder
	proc.Stdout = &stdoutBuf
	proc.Stderr = &stderrBuf

	err := proc.Run()

	if ctx.Err() == context.DeadlineExceeded {
		return -1, stdoutBuf.String(), stderrBuf.String(), true
	}
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return exitErr.ExitCode(), stdoutBuf.String(), stderrBuf.String(), false
		}
		return 1, "", err.Error(), false
	}
	return 0, stdoutBuf.String(), stderrBuf.String(), false
}

func formatDuration(s float64) string {
	if s < 60 {
		return fmt.Sprintf("%.1fs", s)
	}
	if s < 3600 {
		return fmt.Sprintf("%dm %ds", int(s)/60, int(s)%60)
	}
	return fmt.Sprintf("%dh %dm", int(s)/3600, (int(s)%3600)/60)
}

func isKernelsListName(name string) bool {
	return name == "kernelslist" || strings.HasPrefix(name, "kernelslist_")
}

func isStatsName(name string) bool {
	return name == "stats" || strings.HasPrefix(name, "stats_")
}

func traceTimeoutMarkerPath(cfgTraceDir string) string {
	return filepath.Join(cfgTraceDir, "traces", "trace_timeout.json")
}

func traceTimedOut(cfgTraceDir string) bool {
	_, err := os.Stat(traceTimeoutMarkerPath(cfgTraceDir))
	return err == nil
}

func markTraceTimedOut(cfgTraceDir string, timeoutSecs int) error {
	if err := os.RemoveAll(cfgTraceDir); err != nil {
		return err
	}
	tracesDir := filepath.Join(cfgTraceDir, "traces")
	if err := os.MkdirAll(tracesDir, 0755); err != nil {
		return err
	}
	marker := fmt.Sprintf(
		"{\n  \"status\": \"timeout\",\n  \"timeout_seconds\": %d,\n  \"recorded_at\": %q\n}\n",
		timeoutSecs, time.Now().UTC().Format(time.RFC3339),
	)
	return os.WriteFile(traceTimeoutMarkerPath(cfgTraceDir), []byte(marker), 0644)
}

func traceNameFromKernelsListRef(ref string) string {
	ref = filepath.Base(strings.TrimSpace(ref))
	switch {
	case strings.HasSuffix(ref, ".trace.zsim"):
		return strings.TrimSuffix(ref, ".trace.zsim") + ".trace.zst"
	case strings.HasSuffix(ref, ".trace.zst"):
		return ref
	default:
		return ""
	}
}

func kernelsListTraceRefs(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var refs []string
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.FieldsFunc(line, func(r rune) bool {
			return r == ',' || r == ' ' || r == '\t'
		})
		for _, field := range fields {
			if name := traceNameFromKernelsListRef(field); name != "" {
				refs = append(refs, name)
			}
		}
	}
	return refs, nil
}

func traceComplete(cfgTraceDir string) (complete bool, exists bool, detail string) {
	tracesDir := filepath.Join(cfgTraceDir, "traces")
	entries, err := os.ReadDir(tracesDir)
	if err != nil {
		if os.IsNotExist(err) {
			return false, false, "missing traces directory"
		}
		return false, true, err.Error()
	}

	kernelsLists := make([]string, 0)
	statsFiles := make(map[string]bool)
	traceFiles := make(map[string]bool)

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		switch {
		case isKernelsListName(name):
			kernelsLists = append(kernelsLists, name)
		case isStatsName(name):
			statsFiles[name] = true
		case strings.HasSuffix(name, ".trace.zst"):
			traceFiles[name] = true
		}
	}

	if len(kernelsLists) == 0 {
		return false, true, "missing kernelslist"
	}
	if len(statsFiles) == 0 {
		return false, true, "missing stats"
	}
	if len(traceFiles) == 0 {
		return false, true, "missing trace files"
	}

	sort.Strings(kernelsLists)
	var lastDetail string
	for _, kernelsList := range kernelsLists {
		suffix := strings.TrimPrefix(kernelsList, "kernelslist")
		if suffix != "" && !statsFiles["stats"+suffix] {
			lastDetail = fmt.Sprintf("%s has no matching stats%s", kernelsList, suffix)
			continue
		}

		refs, err := kernelsListTraceRefs(filepath.Join(tracesDir, kernelsList))
		if err != nil {
			lastDetail = fmt.Sprintf("cannot read %s: %v", kernelsList, err)
			continue
		}
		if len(refs) == 0 {
			lastDetail = fmt.Sprintf("%s has no kernel trace references", kernelsList)
			continue
		}

		missing := ""
		for _, ref := range refs {
			if !traceFiles[ref] {
				missing = ref
				break
			}
		}
		if missing == "" {
			return true, true, "complete"
		}
		lastDetail = fmt.Sprintf("%s references missing %s", kernelsList, missing)
	}

	if lastDetail == "" {
		lastDetail = "no complete trace context"
	}
	return false, true, lastDetail
}

func firstKernelsListPath(cfgTraceDir string) string {
	tracesDir := filepath.Join(cfgTraceDir, "traces")
	defaultPath := filepath.Join(tracesDir, "kernelslist")
	if _, err := os.Stat(defaultPath); err == nil {
		return defaultPath
	}

	entries, err := os.ReadDir(tracesDir)
	if err != nil {
		return defaultPath
	}
	var candidates []string
	for _, entry := range entries {
		if !entry.IsDir() && isKernelsListName(entry.Name()) {
			candidates = append(candidates, entry.Name())
		}
	}
	sort.Strings(candidates)
	if len(candidates) == 0 {
		return defaultPath
	}
	return filepath.Join(tracesDir, candidates[0])
}

// ── Command: run ───────────────────────────────────────────────────────────

type runArgs struct {
	tier            string
	suite           string
	benchmark       string
	reps            string
	db              string
	platform        string
	sweepNonScaling bool
	timeout         string
	scalingValues   string
	dryRun          bool
	continueOnError bool
	verbose         bool
	collectTrace    string
	traceOutput     string
	traceSample     string // PTRACE_SAMPLE spec for live-sampling tracer ("" = full)
	traceSizes      int    // trace only the N smallest scaling values (0 = all)
	postProcess     string
	resumeTrace     bool
	cleanIncomplete bool
}

func cmdRun(args runArgs) int {
	repoRoot := findRepoRoot()
	tier := parseTier(args.tier)
	reps := DefaultReps
	if args.reps != "" {
		if r, err := strconv.Atoi(args.reps); err == nil {
			reps = r
		}
	}
	dbPath := args.db
	if dbPath == "" {
		dbPath = DefaultDB
	}
	plat := args.platform
	if plat == "" {
		plat = detectPlatform()
	}
	timeoutSecs := DefaultTimeout
	if args.timeout != "" {
		if t, err := strconv.Atoi(args.timeout); err == nil {
			timeoutSecs = t
		}
	}

	// Parse scaling override values
	var scalingOverride []interface{}
	if args.scalingValues != "" {
		for _, v := range strings.Split(args.scalingValues, ",") {
			v = strings.TrimSpace(v)
			if i, err := strconv.Atoi(v); err == nil {
				scalingOverride = append(scalingOverride, float64(i))
			} else if f, err := strconv.ParseFloat(v, 64); err == nil {
				scalingOverride = append(scalingOverride, f)
			} else {
				scalingOverride = append(scalingOverride, v)
			}
		}
	}

	benchmarks := discoverBenchmarks(repoRoot, tier, args.suite, args.benchmark)
	if len(benchmarks) == 0 {
		eprint("No benchmarks found matching filters.")
		return ExitNoBench
	}

	// Count total runs
	totalRuns := 0
	for _, b := range benchmarks {
		configs := generateConfigs(b, args.sweepNonScaling, scalingOverride)
		totalRuns += len(configs) * reps
	}

	info := collectPlatformInfo(plat)

	eprint("=== GPU Benchmark Suite ===")
	eprintf("Platform: %s (%s)\n", strings.ToUpper(plat), info.GPUModel)
	eprintf("Database: %s\n", dbPath)
	eprintf("Repetitions: %d\n", reps)
	eprintf("Benchmarks: %d\n", len(benchmarks))
	eprintf("Total runs: %d\n\n", totalRuns)

	if args.dryRun {
		eprint("--- DRY RUN (no execution) ---\n")
		for idx, b := range benchmarks {
			spName := b.ScalingParam.Name
			configs := generateConfigs(b, args.sweepNonScaling, scalingOverride)
			cmd := getBenchCommand(b, plat)
			eprintf("[%3d/%d] tier%d/%s (%s)\n",
				idx+1, len(benchmarks), b.Tier, b.Benchmark, getLanguage(b))
			eprintf("  Scaling: %s (%d configs × %d reps = %d runs)\n",
				spName, len(configs), reps, len(configs)*reps)
			eprintf("  Command: %s\n", strings.Join(cmd, " "))

			// Collect unique scaling values
			svSet := make(map[string]interface{})
			for _, c := range configs {
				svSet[formatValue(c.ScalingValue)] = c.ScalingValue
			}
			var svVals []interface{}
			for _, v := range svSet {
				svVals = append(svVals, v)
			}
			sortValues(svVals)

			if len(svVals) <= 10 {
				parts := make([]string, len(svVals))
				for i, v := range svVals {
					parts[i] = formatValue(v)
				}
				eprintf("  Values:  %s\n", strings.Join(parts, ", "))
			} else {
				eprintf("  Values:  %s ... %s (%d values)\n",
					formatValue(svVals[0]),
					formatValue(svVals[len(svVals)-1]),
					len(svVals))
			}
			eprint()
		}
		eprintf("Total: %d runs across %d benchmarks\n", totalRuns, len(benchmarks))
		return ExitOK
	}

	// Trace collection mode: skip SQLite writes, inject LD_PRELOAD env vars.
	// Uses a higher default timeout (NVBit adds 100-1000x overhead).
	// When a results DB exists, only traces configs that have status='ok' runs.
	if args.collectTrace != "" {
		traceOutput := args.traceOutput
		if traceOutput == "" {
			traceOutput = "./traces"
		}
		// Use trace-specific timeout unless user explicitly set --timeout
		if args.timeout == "" {
			timeoutSecs = DefaultTraceTimeout
		}

		// Open DB read-only to discover which configs have successful runs.
		var dbConfigs map[string]map[string]bool // benchmark -> set of scaling_param_values
		if _, err := os.Stat(dbPath); err == nil {
			db, err := sql.Open("sqlite3", dbPath+"?mode=ro")
			if err == nil {
				defer db.Close()
				dbConfigs = loadSuccessfulConfigs(db)
			}
		}

		eprint("=== Trace Collection Mode ===")
		eprintf("Tracer:      %s\n", args.collectTrace)
		eprintf("Output dir:  %s\n", traceOutput)
		eprintf("Timeout:     %ds\n", timeoutSecs)
		if args.traceSample != "" {
			eprintf("Sampling:    PTRACE_SAMPLE=%s (live block sampling)\n", args.traceSample)
		}
		if args.traceSizes > 0 {
			eprintf("Sizes:       %d smallest scaling values per benchmark\n", args.traceSizes)
		}
		if args.resumeTrace {
			eprint("Resume:      enabled (skipping complete trace configs)")
			if args.cleanIncomplete {
				eprint("Cleanup:     enabled (removing incomplete config dirs before retry)")
			}
		}
		if dbConfigs != nil {
			eprintf("DB guidance: %s (only tracing configs with successful runs)\n", dbPath)
		} else {
			eprint("DB guidance: none (tracing all configs)")
		}
		if args.postProcess != "" {
			eprintf("Post-process: %s\n", args.postProcess)
		}
		eprint("NOTE: SQLite writes are disabled in trace collection mode.")

		setupInterruptHandler()
		totalCaptured, totalFailed, totalDBSkipped := 0, 0, 0
		totalResumeSkipped, totalRetried, totalTimedOutSkipped := 0, 0, 0
		startTime := time.Now()

		for idx, b := range benchmarks {
			if interrupted {
				break
			}
			spName := b.ScalingParam.Name
			configs := generateConfigs(b, args.sweepNonScaling, scalingOverride)

			// Filter configs to only those with successful DB runs
			if dbConfigs != nil {
				if benchSet, ok := dbConfigs[b.Benchmark]; ok {
					var filtered []configPair
					for _, cfg := range configs {
						if benchSet[formatValue(cfg.ScalingValue)] {
							filtered = append(filtered, cfg)
						}
					}
					skipped := len(configs) - len(filtered)
					totalDBSkipped += skipped
					configs = filtered
				} else {
					// No successful runs for this benchmark at all
					totalDBSkipped += len(configs)
					eprintf("[%3d/%d] tier%d/%s — skipped (no successful runs in DB)\n",
						idx+1, len(benchmarks), b.Tier, b.Benchmark)
					continue
				}
			}

			if args.traceSizes > 0 {
				configs = smallestScalingConfigs(configs, args.traceSizes)
			}

			if len(configs) == 0 {
				continue
			}

			eprintf("[%3d/%d] tier%d/%s\n", idx+1, len(benchmarks), b.Tier, b.Benchmark)
			eprintf("  Scaling: %s (%d configs x %d reps = %d runs)\n",
				spName, len(configs), reps, len(configs)*reps)

			bp, bf, bs, br, bt := 0, 0, 0, 0, 0
			cmd := getBenchCommand(b, plat)

			for _, cfg := range configs {
				if interrupted {
					break
				}

				configDir := traceConfigDir(spName, cfg.ScalingValue, cfg.NSDict)
				cfgLabel := filepath.Join(b.Benchmark, configDir)
				cfgTraceDir := filepath.Join(traceOutput, b.Benchmark, configDir)

				retryDetail := ""
				if args.resumeTrace {
					if traceTimedOut(cfgTraceDir) {
						eprintf("    %s skip (previous timeout)\n", cfgLabel)
						bt++
						continue
					}
					complete, exists, detail := traceComplete(cfgTraceDir)
					if complete {
						eprintf("    %s skip (trace complete)\n", cfgLabel)
						bs++
						continue
					}
					if exists {
						retryDetail = detail
						br++
						if args.cleanIncomplete {
							if err := os.RemoveAll(cfgTraceDir); err != nil {
								eprintf("Error cleaning incomplete trace dir %s: %v\n", cfgTraceDir, err)
								return ExitDB
							}
						}
					}
				}

				if err := os.MkdirAll(cfgTraceDir, 0755); err != nil {
					eprintf("Error creating trace dir %s: %v\n", cfgTraceDir, err)
					return ExitDB
				}
				if retryDetail != "" {
					cleanup := ""
					if args.cleanIncomplete {
						cleanup = "; cleaned"
					}
					eprintf("    %s retry incomplete (%s%s)\n",
						cfgLabel, retryDetail, cleanup)
				}

				cfgCaptured := false
				cfgTimedOut := false
				for rep := 1; rep <= reps; rep++ {
					if interrupted {
						break
					}
					env := buildEnv(spName, cfg.ScalingValue, cfg.NSDict)
					env = append(env,
						fmt.Sprintf("LD_PRELOAD=%s", args.collectTrace),
						"USER_DEFINED_FOLDERS=1",
						fmt.Sprintf("TRACES_FOLDER=%s", cfgTraceDir),
						// Trace collection wants one launch per kernel, not the
						// timing harness's warmup + repeated iterations: nvbit
						// slows every launch, and duplicate launches multiply
						// trace size and timeout risk without adding coverage.
						"BENCH_PARAM_num_warmup=0",
						"BENCH_PARAM_num_iters=1",
					)
					if args.traceSample != "" {
						env = append(env, fmt.Sprintf("PTRACE_SAMPLE=%s", args.traceSample))
					}
					eprintf("    %s rep %d/%d  ", cfgLabel, rep, reps)

					ec, _, stderrStr, timedOut := runBenchmarkProc(
						cmd, env, timeoutSecs, args.verbose)

					var status string
					if timedOut {
						status = "timeout"
					} else if ec != 0 {
						status = "error"
					} else {
						status = "ok"
					}

					if (status != "ok" || args.verbose) && stderrStr != "" {
						for _, sl := range strings.Split(stderrStr, "\n") {
							if sl != "" {
								eprintf("      [stderr] %s\n", sl)
							}
						}
					}

					if status == "ok" {
						eprint("trace captured")
						bp++
						cfgCaptured = true
					} else if status == "timeout" {
						eprintf("timeout after %ds\n", timeoutSecs)
						if err := markTraceTimedOut(cfgTraceDir, timeoutSecs); err != nil {
							eprintf("Error marking timeout trace dir %s: %v\n", cfgTraceDir, err)
							return ExitDB
						}
						eprintf("      removed trace data; wrote %s\n", traceTimeoutMarkerPath(cfgTraceDir))
						bf++
						cfgTimedOut = true
					} else {
						eprintf("exit code %d\n", ec)
						bf++
					}

					if cfgTimedOut || (!args.continueOnError && bf > 0 && status != "ok") {
						break
					}
				}

				if args.postProcess != "" && cfgCaptured && !cfgTimedOut {
					kernelsList := firstKernelsListPath(cfgTraceDir)
					eprintf("  Post-processing: %s %s\n", args.postProcess, kernelsList)
					ppCtx, ppCancel := context.WithTimeout(context.Background(), 300*time.Second)
					ppCmd := exec.CommandContext(ppCtx, args.postProcess, kernelsList)
					ppCmd.Stdout = os.Stdout
					ppCmd.Stderr = os.Stderr
					if err := ppCmd.Run(); err != nil {
						eprintf("  Post-process error: %v\n", err)
					}
					ppCancel()
				}

				if !args.continueOnError && bf > 0 {
					break
				}
			}

			totalCaptured += bp
			totalFailed += bf
			totalResumeSkipped += bs
			totalRetried += br
			totalTimedOutSkipped += bt
			sym := "ok"
			if bf > 0 {
				sym = "FAIL"
			}
			eprintf("  [%s] captured=%d failed=%d skipped_complete=%d skipped_timeout=%d retried_incomplete=%d traces: %s\n\n",
				sym, bp, bf, bs, bt, br, filepath.Join(traceOutput, b.Benchmark))
		}

		eprint("=== Trace Collection Summary ===")
		eprintf("  Attempted:          %d runs\n", totalCaptured+totalFailed)
		eprintf("  Captured:           %d\n", totalCaptured)
		eprintf("  Failed:             %d\n", totalFailed)
		if totalResumeSkipped > 0 {
			eprintf("  Skipped complete:   %d configs\n", totalResumeSkipped)
		}
		if totalTimedOutSkipped > 0 {
			eprintf("  Skipped timeout:    %d configs\n", totalTimedOutSkipped)
		}
		if totalRetried > 0 {
			eprintf("  Retried incomplete: %d configs\n", totalRetried)
		}
		if totalDBSkipped > 0 {
			eprintf("  DB-skipped:         %d configs (no successful runs in DB)\n", totalDBSkipped)
		}
		eprintf("  Time:    %s\n", formatDuration(time.Since(startTime).Seconds()))
		eprintf("  Output:  %s\n", traceOutput)

		if interrupted {
			return ExitInt
		}
		if totalFailed > 0 {
			return ExitSomeFailed
		}
		return ExitOK
	}

	// Open DB, create session
	db := openDatabase(dbPath)
	defer db.Close()
	platformID := getOrCreatePlatform(db, info)
	sessionID := createSession(db, platformID, getCmdLine(), reps,
		args.tier, args.suite, args.benchmark)

	setupInterruptHandler()

	totalPassed, totalFailed := 0, 0
	startTime := time.Now()

	defer func() {
		ss := "completed"
		if interrupted {
			ss = "interrupted"
		} else if totalFailed > 0 {
			ss = "failed"
		}
		finishSession(db, sessionID, ss)
		eprint("=== Summary ===")
		eprintf("  Total:   %d runs\n", totalPassed+totalFailed)
		eprintf("  Passed:  %d\n", totalPassed)
		eprintf("  Failed:  %d\n", totalFailed)
		eprintf("  Time:    %s\n", formatDuration(time.Since(startTime).Seconds()))
		eprintf("  Database: %s\n", dbPath)
	}()

	for idx, b := range benchmarks {
		if interrupted {
			break
		}
		spName := b.ScalingParam.Name
		configs := generateConfigs(b, args.sweepNonScaling, scalingOverride)
		eprintf("[%3d/%d] tier%d/%s\n", idx+1, len(benchmarks), b.Tier, b.Benchmark)
		eprintf("  Scaling: %s (%d configs x %d reps = %d runs)\n",
			spName, len(configs), reps, len(configs)*reps)

		bp, bf := 0, 0
		cmd := getBenchCommand(b, plat)

		for _, cfg := range configs {
			if interrupted {
				break
			}
			for rep := 1; rep <= reps; rep++ {
				if interrupted {
					break
				}
				env := buildEnv(spName, cfg.ScalingValue, cfg.NSDict)
				eprintf("    %s=%-12s rep %d/%d  ",
					spName, formatValue(cfg.ScalingValue), rep, reps)

				ec, stdoutStr, stderrStr, timedOut := runBenchmarkProc(
					cmd, env, timeoutSecs, args.verbose)

				kernels, summary, protoErrors := parseOutput(stdoutStr)

				var status, errMsg string
				if timedOut {
					status = "timeout"
					errMsg = fmt.Sprintf("Timed out after %ds", timeoutSecs)
				} else if ec != 0 {
					status = "error"
					if len(protoErrors) > 0 && protoErrors[0].Message != "" {
						errMsg = protoErrors[0].Message
					} else {
						errMsg = fmt.Sprintf("Exit code %d", ec)
					}
				} else {
					status = "ok"
				}

				if (status != "ok" || args.verbose) && stderrStr != "" {
					for _, sl := range strings.Split(stderrStr, "\n") {
						if sl != "" {
							eprintf("      [stderr] %s\n", sl)
						}
					}
				}

				if status != "ok" {
					hint := diagnoseBenchError(stderrStr, b.Benchmark)
					if hint != "" {
						eprintf("    hint: %s\n", hint)
					}
				}

				if status != "ok" && stderrStr != "" {
					truncatedStderr := stderrStr
					if len(truncatedStderr) > 1000 {
						truncatedStderr = truncatedStderr[:1000] + "... (truncated)"
					}
					if errMsg != "" {
						errMsg = errMsg + "\n--- stderr ---\n" + truncatedStderr
					} else {
						errMsg = truncatedStderr
					}
				}

				var totalTime sql.NullFloat64
				if summary != nil && summary.TotalTimeMS != nil {
					totalTime = sql.NullFloat64{Float64: *summary.TotalTimeMS, Valid: true}
				}

				func() {
					// One transaction per run: per-iteration timing writes
					// 7-28 kernel rows per run, and per-row autocommit would
					// fsync the WAL for each.
					tx, err := db.Begin()
					if err != nil {
						eprintf("DB error: %v\n", err)
						return
					}
					defer func() { _ = tx.Rollback() }()
					res, err := tx.Exec(
						"INSERT INTO runs (session_id,tier,suite,benchmark,"+
							"scaling_param_name,scaling_param_value,"+
							"non_scaling_params_json,repetition,status,"+
							"error_message,total_time_ms,exit_code,finished_at) "+
							"VALUES (?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))",
						sessionID, b.Tier, b.Suite, b.Benchmark,
						spName, formatValue(cfg.ScalingValue),
						toJSONString(cfg.NSDict), rep,
						status, nullString(errMsg),
						totalTime, ec,
					)
					if err != nil {
						eprintf("DB error: %v\n", err)
						return
					}
					runID, _ := res.LastInsertId()
					for _, kr := range kernels {
						var hostTime sql.NullFloat64
						if kr.HostTimeMS != nil {
							hostTime = sql.NullFloat64{Float64: *kr.HostTimeMS, Valid: true}
						}
						var iteration sql.NullInt64
						if kr.Iteration != nil {
							iteration = sql.NullInt64{Int64: int64(*kr.Iteration), Valid: true}
						}
						_, _ = tx.Exec(
							"INSERT INTO kernel_results (run_id,kernel_name,time_ms,ordinal,"+
								"host_time_ms,iteration) VALUES (?,?,?,?,?,?)",
							runID, kr.Name, kr.TimeMS, kr.Ordinal, hostTime, iteration,
						)
					}
					if summary != nil {
						for _, m := range summary.Metrics {
							_, _ = tx.Exec(
								"INSERT INTO metrics (run_id,metric_name,metric_value) "+
									"VALUES (?,?,?)",
								runID, m.Name, m.Value,
							)
						}
					}
					if err := tx.Commit(); err != nil {
						eprintf("DB error: %v\n", err)
					}
				}()

				if status == "ok" {
					if totalTime.Valid {
						eprintf("%.3fms  ok\n", totalTime.Float64)
					} else {
						eprint("ok")
					}
					bp++
				} else if status == "timeout" {
					eprint("TIMEOUT")
					bf++
				} else {
					if errMsg != "" {
						eprintf("FAIL (%s)\n", errMsg)
					} else {
						eprint("FAIL (error)")
					}
					bf++
				}

				if !args.continueOnError && bf > 0 && status != "ok" {
					break
				}
			}
			if !args.continueOnError && bf > 0 {
				break
			}
		}

		totalPassed += bp
		totalFailed += bf
		sym := "ok"
		if bf > 0 {
			sym = "FAIL"
		}
		eprintf("  [%s] %d/%d passed (%d failed)\n\n", sym, bp, bp+bf, bf)
	}

	if interrupted {
		return ExitInt
	}
	if totalFailed > 0 {
		return ExitSomeFailed
	}
	return ExitOK
}

// ── Diagnostics ──────────────────────────────────────────────────────────

func diagnoseBenchError(stderrStr string, benchmark string) string {
	lower := strings.ToLower(stderrStr)

	// Binary not found
	if strings.Contains(lower, "no such file or directory") ||
		strings.Contains(lower, "executable file not found") ||
		strings.Contains(lower, "not found") && strings.Contains(lower, "exec") {
		return fmt.Sprintf("Binary not found. Run `gpubench compile --benchmark %s` first.", benchmark)
	}

	// Common CUDA/GPU errors
	if strings.Contains(lower, "cuda error") || strings.Contains(lower, "cudaerror") {
		return "CUDA error detected. Check GPU driver installation and compatibility."
	}
	if strings.Contains(lower, "no cuda-capable device") || strings.Contains(lower, "no cuda") {
		return "No CUDA-capable GPU detected. Verify GPU is present and drivers are installed."
	}
	if strings.Contains(lower, "out of memory") || strings.Contains(lower, "alloc failed") {
		return "GPU out of memory. Try reducing problem size or freeing GPU memory."
	}
	if strings.Contains(lower, "libcudnn not found") || strings.Contains(lower, "libcudnn") {
		return "cuDNN library not found. Install cuDNN or check LD_LIBRARY_PATH."
	}
	if strings.Contains(lower, "libcublas") || strings.Contains(lower, "libcufft") {
		return "CUDA library missing. Check CUDA toolkit installation."
	}

	return ""
}

// ── Helpers ────────────────────────────────────────────────────────────────

func toJSONString(m map[string]interface{}) string {
	data, err := json.Marshal(m)
	if err != nil {
		return "{}"
	}
	return string(data)
}

// traceConfigDir builds a directory name encoding the full parameter config.
// Example: "N_1024" or "M_512__TILE_16__ALGO_naive".
func traceConfigDir(
	spName string, spValue interface{}, nsDict map[string]interface{},
) string {
	parts := []string{fmt.Sprintf("%s_%s", spName, formatValue(spValue))}
	if len(nsDict) > 0 {
		keys := make([]string, 0, len(nsDict))
		for k := range nsDict {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			parts = append(parts, fmt.Sprintf("%s_%s", k, formatValue(nsDict[k])))
		}
	}
	return strings.Join(parts, "__")
}

func nullString(s string) sql.NullString {
	if s == "" {
		return sql.NullString{}
	}
	return sql.NullString{String: s, Valid: true}
}

// loadSuccessfulConfigs queries the DB for all (benchmark, scaling_value)
// pairs that have at least one successful run. Returns a map of
// benchmark -> set of scaling_param_value strings.
func loadSuccessfulConfigs(db *sql.DB) map[string]map[string]bool {
	rows, err := db.Query(
		`SELECT DISTINCT benchmark, scaling_param_value
		 FROM runs WHERE status = 'ok'`)
	if err != nil {
		eprintf("Warning: could not query DB for successful configs: %v\n", err)
		return nil
	}
	defer rows.Close()

	result := make(map[string]map[string]bool)
	count := 0
	for rows.Next() {
		var bench, spVal string
		if err := rows.Scan(&bench, &spVal); err != nil {
			continue
		}
		if result[bench] == nil {
			result[bench] = make(map[string]bool)
		}
		result[bench][spVal] = true
		count++
	}
	eprintf("  Loaded %d successful data points across %d benchmarks from DB\n", count, len(result))
	return result
}
