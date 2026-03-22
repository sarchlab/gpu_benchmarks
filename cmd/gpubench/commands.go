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
		} else if err != nil {
			eprint("✗")
			failed++
			failures = append(failures, b.Benchmark)
			if verbose {
				eprint(stderrBuf.String())
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
		eprintf("  Scaling: %s (%d configs × %d reps = %d runs)\n",
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
				if args.verbose && stderrStr != "" {
					for _, sl := range strings.Split(stderrStr, "\n") {
						if sl != "" {
							eprintf("      [stderr] %s\n", sl)
						}
					}
				}

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

				var totalTime sql.NullFloat64
				if summary != nil && summary.TotalTimeMS != nil {
					totalTime = sql.NullFloat64{Float64: *summary.TotalTimeMS, Valid: true}
				}

				// Insert into DB
				func() {
					res, err := db.Exec(
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
						_, _ = db.Exec(
							"INSERT INTO kernel_results (run_id,kernel_name,time_ms,ordinal) "+
								"VALUES (?,?,?,?)",
							runID, kr.Name, kr.TimeMS, kr.Ordinal,
						)
					}
					if summary != nil {
						for _, m := range summary.Metrics {
							_, _ = db.Exec(
								"INSERT INTO metrics (run_id,metric_name,metric_value) "+
									"VALUES (?,?,?)",
								runID, m.Name, m.Value,
							)
						}
					}
				}()

				if status == "ok" {
					if totalTime.Valid {
						eprintf("%.3fms  ✓\n", totalTime.Float64)
					} else {
						eprint("✓")
					}
					bp++
				} else if status == "timeout" {
					eprint("✗ (timeout)")
					bf++
				} else {
					if errMsg != "" {
						eprintf("✗ (%s)\n", errMsg)
					} else {
						eprint("✗ (error)")
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
		sym := "✓"
		if bf > 0 {
			sym = "✗"
		}
		eprintf("  %s %d/%d passed (%d failed)\n\n", sym, bp, bp+bf, bf)
	}

	if interrupted {
		return ExitInt
	}
	if totalFailed > 0 {
		return ExitSomeFailed
	}
	return ExitOK
}

// ── Helpers ────────────────────────────────────────────────────────────────

func toJSONString(m map[string]interface{}) string {
	data, err := json.Marshal(m)
	if err != nil {
		return "{}"
	}
	return string(data)
}

func nullString(s string) sql.NullString {
	if s == "" {
		return sql.NullString{}
	}
	return sql.NullString{String: s, Valid: true}
}
