package main

import (
	"context"
	"encoding/csv"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// rocprof integration.
//
// When --rocprof is set, the benchmark is run under rocprof and the per-kernel
// time reported to the database comes from rocprof's HARDWARE timestamps
// (command-processor begin/end), not the benchmark's own hipEvent timing. This
// is GPU-only kernel time -- it excludes host launch overhead and is the
// faithful measurement for launch-dominated benchmarks.
//
// The benchmark still runs normally and its stdout (kernel names + params) is
// captured as before; only the time_ms is overridden by rocprof's DurationNs
// for the matching kernel.
//
// rocprof's CLI and CSV layout differ across versions, so:
//   - the invocation is a template (--rocprof-cmd), {OUT} = the per-dispatch CSV
//     path and {OUTDIR} = its directory;
//   - parseRocprofCSV is tolerant of column names (KernelName/Kernel_Name/Name
//     and DurationNs, or Begin/End timestamps).
// Default targets rocprof v1 (`rocprof --basenames on --stats -o <csv> <app>`).
// For rocprofv3 pass e.g.
//   --rocprof-cmd 'rocprofv3 --kernel-trace --output-format csv -d {OUTDIR} -- '

const defaultRocprofCmd = "rocprof --basenames on --stats -o {OUT}"

// rocprofAvailable reports whether a rocprof binary is on PATH.
func rocprofAvailable() bool {
	for _, name := range []string{"rocprof", "rocprofv3", "rocprofv2"} {
		if _, err := exec.LookPath(name); err == nil {
			return true
		}
	}
	return false
}

// runUnderRocprof runs benchCmd under rocprof. It returns the benchmark's own
// stdout/stderr (for kernel names + params) and, separately, the per-kernel GPU
// duration in ms parsed from rocprof's hardware timestamps. For each kernel the
// LAST dispatch is used -- i.e. the timed launch that follows the warmups.
//
// NOTE: "last dispatch" is correct for the single-timed-launch microbenchmarks.
// Benchmarks that launch a kernel multiple times inside the timed region (e.g.
// iterated stencils) would need the per-region sum; mark those with roctx ranges
// and extend this parser if/when they are profiled with rocprof.
func runUnderRocprof(
	benchCmd, env []string, timeoutSecs int, rocprofTmpl string,
) (ec int, stdout, stderr string, timedOut bool, perKernelMs map[string]float64) {
	outDir, err := os.MkdirTemp("", "gpubench-rocprof-")
	if err != nil {
		return 1, "", "rocprof: " + err.Error(), false, nil
	}
	defer os.RemoveAll(outDir)
	outCSV := filepath.Join(outDir, "results.csv")

	tmpl := rocprofTmpl
	if tmpl == "" {
		tmpl = defaultRocprofCmd
	}
	tmpl = strings.ReplaceAll(tmpl, "{OUT}", outCSV)
	tmpl = strings.ReplaceAll(tmpl, "{OUTDIR}", outDir)

	full := append(strings.Fields(tmpl), benchCmd...)

	ctx, cancel := context.WithTimeout(
		context.Background(), time.Duration(timeoutSecs)*time.Second)
	defer cancel()

	proc := exec.CommandContext(ctx, full[0], full[1:]...)
	proc.Env = env

	var outBuf, errBuf strings.Builder
	proc.Stdout = &outBuf
	proc.Stderr = &errBuf
	runErr := proc.Run()

	if ctx.Err() == context.DeadlineExceeded {
		return -1, outBuf.String(), errBuf.String(), true, nil
	}

	exitCode := 0
	if runErr != nil {
		if ee, ok := runErr.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
		} else {
			return 1, outBuf.String(),
				errBuf.String() + "\nrocprof: " + runErr.Error(), false, nil
		}
	}

	return exitCode, outBuf.String(), errBuf.String(), false,
		parseRocprofCSV(outCSV)
}

// parseRocprofCSV reads a rocprof per-dispatch CSV and returns, per kernel base
// name, the GPU duration (ms) of its LAST dispatch. Tolerant of column-name
// differences across rocprof versions; falls back to End-Begin if there is no
// explicit duration column. Returns nil if nothing usable is found.
func parseRocprofCSV(path string) map[string]float64 {
	f, err := os.Open(path)
	if err != nil {
		// rocprof v1 may also name it <base>.csv next to the requested file.
		alt := strings.TrimSuffix(path, ".csv") + ".stats.csv"
		if f, err = os.Open(alt); err != nil {
			return nil
		}
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = -1
	rows, err := r.ReadAll()
	if err != nil || len(rows) < 2 {
		return nil
	}

	header := rows[0]
	colOf := func(names ...string) int {
		for i, h := range header {
			hl := strings.ToLower(strings.TrimSpace(h))
			for _, n := range names {
				if hl == strings.ToLower(n) {
					return i
				}
			}
		}
		return -1
	}

	nameCol := colOf("KernelName", "Kernel_Name", "Name", "kernel_name")
	durCol := colOf("DurationNs", "Duration_ns", "duration_ns", "DurationNsec")
	beginCol := colOf("BeginNs", "Start_Timestamp", "BeginNsec", "start_ns")
	endCol := colOf("EndNs", "End_Timestamp", "EndNsec", "end_ns")
	if nameCol < 0 || (durCol < 0 && (beginCol < 0 || endCol < 0)) {
		return nil
	}

	out := map[string]float64{}
	for _, row := range rows[1:] {
		if nameCol >= len(row) {
			continue
		}
		name := kernelBaseName(row[nameCol])

		var durNs float64
		switch {
		case durCol >= 0 && durCol < len(row):
			durNs, _ = strconv.ParseFloat(strings.TrimSpace(row[durCol]), 64)
		case beginCol < len(row) && endCol < len(row):
			b, _ := strconv.ParseFloat(strings.TrimSpace(row[beginCol]), 64)
			e, _ := strconv.ParseFloat(strings.TrimSpace(row[endCol]), 64)
			durNs = e - b
		}

		if name == "" || durNs <= 0 {
			continue
		}
		out[name] = durNs / 1e6 // ns -> ms; last dispatch wins (the timed launch)
	}

	if len(out) == 0 {
		return nil
	}
	return out
}

// kernelBaseName reduces a (possibly demangled) kernel symbol to the bare
// function name so it matches the benchmark's reported kernel name, e.g.
// "smem_bw_no_conflict(float const*, float*, int)" -> "smem_bw_no_conflict".
func kernelBaseName(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '('); i >= 0 {
		s = s[:i]
	}
	if i := strings.LastIndex(s, "::"); i >= 0 {
		s = s[i+2:]
	}
	return strings.TrimSpace(s)
}
