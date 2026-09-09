package main

import (
	"math"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

// PlatformInfo holds detected platform information.
type PlatformInfo struct {
	PlatformType   string   `json:"platform_type"`
	Hostname       string   `json:"hostname"`
	OS             string   `json:"os"`
	CPUModel       *string  `json:"cpu_model"`
	CPUCores       *int     `json:"cpu_cores"`
	RAMGB          *float64 `json:"ram_gb"`
	GPUModel       string   `json:"gpu_model"`
	GPUVramGB      *float64 `json:"gpu_vram_gb"`
	GPUCount       int      `json:"gpu_count"`
	DriverVersion  *string  `json:"driver_version"`
	RuntimeVersion *string  `json:"runtime_version"` // unknown without observing the benchmark runtime
	Compiler       *string  `json:"compiler"`        // installed tool, not necessarily the binary's compiler
	PythonVersion  string   `json:"python_version"`  // stores Go version for schema compat
}

var (
	platformCache *PlatformInfo
	platformOnce  sync.Once
)

// detectPlatform returns "rocm" or "cuda". The hipcc check matches the
// Makefiles' detection, so build and run agree on the platform.
func detectPlatform() string {
	if _, err := os.Stat("/opt/rocm/bin/hipcc"); err == nil {
		return "rocm"
	}
	return "cuda"
}

// runCmd runs a shell command and returns stdout stripped, or empty string on failure.
func runCmd(cmd string) string {
	out, err := exec.Command("sh", "-c", cmd).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func strPtr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

func intPtr(i int) *int {
	return &i
}

func floatPtr(f float64) *float64 {
	return &f
}

// collectPlatformInfo gathers platform info. Caches result.
func collectPlatformInfo(platformType string) *PlatformInfo {
	platformOnce.Do(func() {
		if platformType == "" {
			platformType = detectPlatform()
		}

		hostname, _ := os.Hostname()

		info := &PlatformInfo{
			PlatformType:  platformType,
			Hostname:      hostname,
			GPUModel:      "Unknown GPU",
			GPUCount:      1,
			PythonVersion: runtime.Version(), // Go version stored in python_version field
		}

		// OS detection
		osStr := runCmd("uname -srm")
		if osStr == "" {
			osStr = runtime.GOOS + " " + runtime.GOARCH
		}
		info.OS = osStr

		// CPU/RAM detection
		if runtime.GOOS == "darwin" {
			cpuModel := runCmd("sysctl -n machdep.cpu.brand_string")
			if cpuModel == "" {
				cpuModel = runCmd("/usr/sbin/sysctl -n machdep.cpu.brand_string")
			}
			info.CPUModel = strPtr(cpuModel)

			coresStr := runCmd("sysctl -n hw.logicalcpu")
			if coresStr == "" {
				coresStr = runCmd("/usr/sbin/sysctl -n hw.logicalcpu")
			}
			if cores, err := strconv.Atoi(coresStr); err == nil {
				info.CPUCores = intPtr(cores)
			}

			memStr := runCmd("sysctl -n hw.memsize")
			if memStr == "" {
				memStr = runCmd("/usr/sbin/sysctl -n hw.memsize")
			}
			if mem, err := strconv.ParseInt(memStr, 10, 64); err == nil {
				gb := math.Round(float64(mem)/(1024*1024*1024)*10) / 10
				info.RAMGB = floatPtr(gb)
			}
		} else {
			cpu := runCmd("grep -m1 'model name' /proc/cpuinfo | sed 's/model name.*: //'")
			info.CPUModel = strPtr(cpu)

			coresStr := runCmd("nproc 2>/dev/null")
			if cores, err := strconv.Atoi(coresStr); err == nil {
				info.CPUCores = intPtr(cores)
			}

			memStr := runCmd("grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}'")
			if mem, err := strconv.ParseInt(memStr, 10, 64); err == nil {
				gb := math.Round(float64(mem)/(1024*1024)*10) / 10
				info.RAMGB = floatPtr(gb)
			}
		}

		// GPU detection
		switch platformType {
		case "cuda":
			gpu := runCmd("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1")
			if gpu != "" {
				info.GPUModel = gpu
			}
			vram := runCmd("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1")
			if vram != "" {
				if v, err := strconv.ParseFloat(strings.TrimSpace(vram), 64); err == nil {
					gb := math.Round(v/1024*10) / 10
					info.GPUVramGB = floatPtr(gb)
				}
			}
			drv := runCmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1")
			info.DriverVersion = strPtr(drv)
			nvcc := runCmd("nvcc --version 2>/dev/null | grep 'release'")
			if nvcc != "" {
				info.Compiler = strPtr(strings.TrimSpace(nvcc))
			}
			cnt := runCmd("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l")
			if cnt != "" {
				if c, err := strconv.Atoi(strings.TrimSpace(cnt)); err == nil {
					info.GPUCount = c
				}
			}

		case "rocm":
			gpu := runCmd("rocm-smi --showproductname 2>/dev/null | grep 'Card series' | head -1 | sed 's/.*: //'")
			if gpu != "" {
				info.GPUModel = gpu
			}
			// A ROCm release is not a driver version. Likewise, compiler
			// output cannot identify the runtime loaded by a benchmark.
			// Leave those unobserved versions unknown.
			hipcc := runCmd("hipcc --version 2>/dev/null | head -1")
			if hipcc != "" {
				info.Compiler = strPtr(hipcc)
			}
		}

		platformCache = info
	})
	return platformCache
}
