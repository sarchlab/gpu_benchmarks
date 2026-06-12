package main

import (
	"fmt"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/spf13/cobra"
)

// Exit codes matching Python version.
const (
	ExitOK         = 0
	ExitSomeFailed = 1
	ExitUsage      = 2
	ExitNoBench    = 3
	ExitDB         = 4
	ExitInt        = 130
)

// Default values.
const (
	DefaultReps         = 7
	DefaultDB           = "results.db"
	DefaultTimeout      = 60
	DefaultTraceTimeout = 3600
)

// Global interrupt flag.
var (
	interrupted   bool
	interruptOnce sync.Once
	interruptCh   = make(chan struct{})
)

func setupInterruptHandler() {
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGINT)
	go func() {
		<-c
		interruptOnce.Do(func() {
			interrupted = true
			close(interruptCh)
			eprint("\n⚠  Interrupted — finishing current run, then saving partial results...")
		})
	}()
}

func eprint(a ...interface{}) {
	fmt.Fprintln(os.Stderr, a...)
}

func eprintf(format string, a ...interface{}) {
	fmt.Fprintf(os.Stderr, format, a...)
}

func main() {
	rootCmd := &cobra.Command{
		Use:   "gpubench",
		Short: "GPU Benchmark Suite CLI — compile, run, and analyze GPU benchmarks",
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
			os.Exit(ExitUsage)
		},
	}

	// --- list command ---
	listCmd := &cobra.Command{
		Use:   "list",
		Short: "List available benchmarks",
		RunE: func(cmd *cobra.Command, args []string) error {
			tier, _ := cmd.Flags().GetString("tier")
			suite, _ := cmd.Flags().GetString("suite")
			jsonOut, _ := cmd.Flags().GetBool("json")
			os.Exit(cmdList(tier, suite, jsonOut))
			return nil
		},
	}
	listCmd.Flags().String("tier", "", "Filter by tier (1-5)")
	listCmd.Flags().String("suite", "", "Filter by suite")
	listCmd.Flags().Bool("json", false, "JSON output")

	// --- compile command ---
	compileCmd := &cobra.Command{
		Use:   "compile",
		Short: "Compile C++ benchmarks",
		RunE: func(cmd *cobra.Command, args []string) error {
			tier, _ := cmd.Flags().GetString("tier")
			suite, _ := cmd.Flags().GetString("suite")
			benchmark, _ := cmd.Flags().GetString("benchmark")
			platform, _ := cmd.Flags().GetString("platform")
			jobs, _ := cmd.Flags().GetInt("jobs")
			verbose, _ := cmd.Flags().GetBool("verbose")
			os.Exit(cmdCompile(tier, suite, benchmark, platform, jobs, verbose))
			return nil
		},
	}
	compileCmd.Flags().String("tier", "", "Filter by tier")
	compileCmd.Flags().String("suite", "", "Filter by suite")
	compileCmd.Flags().String("benchmark", "", "Filter by benchmark name")
	compileCmd.Flags().String("platform", "", "Platform: cuda, rocm, metal")
	compileCmd.Flags().Int("jobs", 0, "Parallel jobs")
	compileCmd.Flags().Bool("verbose", false, "Verbose output")

	// --- run command ---
	runCmd := &cobra.Command{
		Use:   "run",
		Short: "Run benchmarks",
		RunE: func(cmd *cobra.Command, args []string) error {
			tier, _ := cmd.Flags().GetString("tier")
			suite, _ := cmd.Flags().GetString("suite")
			benchmark, _ := cmd.Flags().GetString("benchmark")
			reps, _ := cmd.Flags().GetString("reps")
			db, _ := cmd.Flags().GetString("db")
			platform, _ := cmd.Flags().GetString("platform")
			sweepNS, _ := cmd.Flags().GetBool("sweep-non-scaling")
			timeout, _ := cmd.Flags().GetString("timeout")
			scalingValues, _ := cmd.Flags().GetString("scaling-values")
			dryRun, _ := cmd.Flags().GetBool("dry-run")
			continueOnError, _ := cmd.Flags().GetBool("continue-on-error")
			verbose, _ := cmd.Flags().GetBool("verbose")
			collectTrace, _ := cmd.Flags().GetString("collect-trace")
			traceOutput, _ := cmd.Flags().GetString("trace-output")
			postProcess, _ := cmd.Flags().GetString("post-process")
			resumeTrace, _ := cmd.Flags().GetBool("resume-trace")
			cleanIncomplete, _ := cmd.Flags().GetBool("trace-clean-incomplete")
			traceSample, _ := cmd.Flags().GetString("trace-sample")
			traceSizes, _ := cmd.Flags().GetInt("trace-sizes")

			os.Exit(cmdRun(runArgs{
				tier:            tier,
				suite:           suite,
				benchmark:       benchmark,
				reps:            reps,
				db:              db,
				platform:        platform,
				sweepNonScaling: sweepNS,
				timeout:         timeout,
				scalingValues:   scalingValues,
				dryRun:          dryRun,
				continueOnError: continueOnError,
				verbose:         verbose,
				collectTrace:    collectTrace,
				traceOutput:     traceOutput,
				postProcess:     postProcess,
				resumeTrace:     resumeTrace,
				cleanIncomplete: cleanIncomplete,
				traceSample:     traceSample,
				traceSizes:      traceSizes,
			}))
			return nil
		},
	}
	runCmd.Flags().String("tier", "", "Filter by tier")
	runCmd.Flags().String("suite", "", "Filter by suite")
	runCmd.Flags().String("benchmark", "", "Filter by benchmark name")
	runCmd.Flags().String("reps", "", "Repetitions (default: 7)")
	runCmd.Flags().String("db", "", "SQLite file (default: results.db)")
	runCmd.Flags().String("platform", "", "Platform: cuda, rocm, metal")
	runCmd.Flags().Bool("sweep-non-scaling", true, "Sweep non-scaling params (use --sweep-non-scaling=false for defaults only)")
	runCmd.Flags().String("timeout", "", "Timeout secs (default: 60)")
	runCmd.Flags().String("scaling-values", "", "Override values (csv)")
	runCmd.Flags().Bool("dry-run", false, "Dry run")
	runCmd.Flags().Bool("continue-on-error", false, "Continue on error")
	runCmd.Flags().Bool("verbose", false, "Verbose output")
	runCmd.Flags().String("collect-trace", "", "Path to NVBit tracer_tool.so for instruction trace collection")
	runCmd.Flags().String("trace-output", "./traces", "Output directory for collected traces (default: ./traces)")
	runCmd.Flags().String("post-process", "", "Optional post-processing binary to run after trace collection")
	runCmd.Flags().Bool("resume-trace", false, "Skip trace configs that already have complete or timed-out trace outputs")
	runCmd.Flags().Bool("trace-clean-incomplete", false, "When resuming traces, delete incomplete config trace directories before retrying")
	runCmd.Flags().String("trace-sample", "", "PTRACE_SAMPLE spec for the live-sampling tracer (e.g. 'corner'); empty = full tracing")
	runCmd.Flags().Int("trace-sizes", 0, "Trace only the N smallest scaling values per benchmark (0 = all configs)")

	// --- info command ---
	infoCmd := &cobra.Command{
		Use:   "info",
		Short: "Display platform information",
		RunE: func(cmd *cobra.Command, args []string) error {
			jsonOut, _ := cmd.Flags().GetBool("json")
			os.Exit(cmdInfo(jsonOut))
			return nil
		},
	}
	infoCmd.Flags().Bool("json", false, "JSON output")

	// --- validate command ---
	validateCmd := &cobra.Command{
		Use:   "validate",
		Short: "Validate params.json files",
		RunE: func(cmd *cobra.Command, args []string) error {
			tier, _ := cmd.Flags().GetString("tier")
			suite, _ := cmd.Flags().GetString("suite")
			benchmark, _ := cmd.Flags().GetString("benchmark")
			os.Exit(cmdValidate(tier, suite, benchmark))
			return nil
		},
	}
	validateCmd.Flags().String("tier", "", "Filter by tier")
	validateCmd.Flags().String("suite", "", "Filter by suite")
	validateCmd.Flags().String("benchmark", "", "Filter by benchmark name")

	// --- export command ---
	exportCmd := &cobra.Command{
		Use:   "export",
		Short: "Export benchmark results as JSON",
		RunE: func(cmd *cobra.Command, args []string) error {
			db, _ := cmd.Flags().GetString("db")
			format, _ := cmd.Flags().GetString("format")
			session, _ := cmd.Flags().GetString("session")
			os.Exit(cmdExport(db, format, session))
			return nil
		},
	}
	exportCmd.Flags().String("db", DefaultDB, "SQLite database file")
	exportCmd.Flags().String("format", "json", "Output format (json)")
	exportCmd.Flags().String("session", "", "Export specific session ID")

	rootCmd.AddCommand(listCmd, compileCmd, runCmd, infoCmd, validateCmd, exportCmd)

	if err := rootCmd.Execute(); err != nil {
		os.Exit(ExitUsage)
	}
}
