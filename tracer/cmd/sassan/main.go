// Command sassan is an offline SASS analyzer: it classifies each kernel's memory
// accesses (affine vs indirect), branches (index- vs data-determined), and
// cross-launch determinism, to tell a sampling tracer what it can synthesize from
// indices versus what it must capture.
//
//	sassan <cubin-or-host-binary>   # runs cuobjdump -sass and analyzes
//	sassan <file.sass>              # analyzes an existing disassembly
package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/sarchlab/gpu_benchmarks/tracer/sass"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: sassan <binary|file.sass>")
		os.Exit(2)
	}
	text, err := loadSASS(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	kernels := sass.ParseKernels(text)
	if len(kernels) == 0 {
		fmt.Fprintln(os.Stderr, "no kernels found (need cuobjdump -sass output)")
		os.Exit(1)
	}
	for _, k := range kernels {
		a := k.Analyze()
		fmt.Printf("══ %s\n", short(k.Name))
		fmt.Printf("   %d instructions\n", len(k.Insts))
		fmt.Printf("   verdict: %s\n", a.Verdict())

		fmt.Printf("   memory accesses:\n")
		for _, m := range a.Mem {
			fmt.Printf("     /*%04x*/ %-16s tier=%-8s base=%-6s%s\n",
				m.Addr, m.Op, m.Tier, m.Base, args(m.Args))
		}
		fmt.Printf("   branches:\n")
		for _, b := range a.Branches {
			loop := ""
			if b.BackEdge {
				loop = "  [loop back-edge]"
			}
			fmt.Printf("     /*%04x*/ @%-4s -> 0x%-4x  %-8s%s\n", b.Addr, b.Pred, b.Target, b.Class, loop)
		}
		fmt.Println()
	}
}

func loadSASS(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err == nil && strings.Contains(string(b), "Function :") {
		return string(b), nil // already a disassembly
	}
	out, err := exec.Command("cuobjdump", "-sass", path).Output()
	if err != nil {
		return "", fmt.Errorf("cuobjdump -sass %s: %w", path, err)
	}
	return string(out), nil
}

// args formats constant-bank offsets as arg indices (sm_90 params start at 0x210).
func args(offs []uint64) string {
	if len(offs) == 0 {
		return ""
	}
	var s []string
	for _, o := range offs {
		if o >= 0x210 {
			s = append(s, fmt.Sprintf("arg%d", (o-0x210)/8))
		} else {
			s = append(s, fmt.Sprintf("c[0x%x]", o))
		}
	}
	return "  <= " + strings.Join(s, ",")
}

func short(name string) string {
	if i := strings.LastIndex(name, "("); i >= 0 {
		name = name[:i]
	}
	return name
}
