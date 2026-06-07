#!/usr/bin/env python3
"""P0 analyzer: measure how synthesizable Accel-Sim traces are.

For each kernel trace it reports:
  - mem address-mode histogram (base_stride / base_delta / list_all)
  - control-flow regularity (distinct per-warp PC sequences)
  - per-static-mem-PC affine fit: base_addr = c0 + c1*block_lin + c2*warp + c3*occ
  - a model-tier verdict and projected size/walltime savings

Usage: ptrace_analyze.py <config-traces-dir> [more dirs...]
A traces dir contains kernelslist_*, stats_*, and kernel-*.trace.zst files.
"""
import glob
import os
import sys

import numpy as np

from ptrace_decode import decode_trace_zst

MODE_NAMES = {0: "list_all", 1: "base_stride", 2: "base_delta"}


def popcount(mask_hex):
    return bin(int(mask_hex, 16)).count("1")


def parse_header(header_text):
    info = {}
    for line in header_text.splitlines():
        if line.startswith("-grid dim"):
            info["grid"] = tuple(int(x) for x in line.split("(")[1].split(")")[0].split(","))
        elif line.startswith("-block dim"):
            info["block"] = tuple(int(x) for x in line.split("(")[1].split(")")[0].split(","))
        elif line.startswith("-kernel name"):
            info["name"] = line.split("=", 1)[1].strip()
    return info


def parse_inst(line):
    """Return dict for one instruction line, or None for non-inst lines."""
    t = line.split()
    if not t:
        return None
    # PC is 4 hex digits; instruction lines start with it.
    try:
        pc = int(t[0], 16)
    except ValueError:
        return None
    if len(t) < 5:
        return None
    mask = t[1]
    i = 2
    dest_num = int(t[i]); i += 1
    i += dest_num  # skip dest reg tokens
    opcode = t[i]; i += 1
    src_num = int(t[i]); i += 1
    i += src_num   # skip src reg tokens
    width_tok = t[i]; i += 1
    imm = int(t[-1])
    rec = {"pc": pc, "mask": mask, "opcode": opcode, "is_mem": False,
           "mode": None, "base": None, "stride": None, "imm": imm}
    if width_tok != "0":
        rec["is_mem"] = True
        rec["width"] = int(width_tok)
        mode = int(t[i]); i += 1
        rec["mode"] = mode
        if mode == 1:  # base_stride: base stride [imm]
            rec["base"] = int(t[i], 16)
            rec["stride"] = int(t[i + 1])
        else:          # list_all / base_delta: base payload... [imm]
            rec["base"] = int(t[i], 16)
            rec["payload"] = t[i + 1:-1]
    return rec


def iter_warps(blocks):
    """Yield (block_coords, warp_idx, [inst lines]) from decoded block texts."""
    for b in blocks:
        text = b.decode("utf-8", "replace")
        coords = None
        warp = None
        insts = []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("thread block ="):
                coords = tuple(int(x) for x in s.split("=", 1)[1].split(","))
            elif s.startswith("warp ="):
                if warp is not None:
                    yield coords, warp, insts
                warp = int(s.split("=", 1)[1])
                insts = []
            elif s.startswith("insts ="):
                continue
            elif s and not s.startswith("#") and warp is not None:
                insts.append(line)
        if warp is not None:
            yield coords, warp, insts


def affine_exact(X, y):
    """True iff y is EXACTLY an affine function of the columns of X (+intercept).

    Uses exact rational Gaussian elimination on the augmented system [X|1 | y];
    y is affine iff that system is consistent. Float lstsq is unusable here
    because addresses are ~2^47 and exceed float64's exact-integer range.
    """
    from fractions import Fraction
    n = len(y)
    if n <= 1:
        return True
    d = len(X[0])
    # Augmented matrix rows: [x0..x_{d-1}, 1, y]
    M = [[Fraction(int(X[i][j])) for j in range(d)] + [Fraction(1), Fraction(int(y[i]))]
         for i in range(n)]
    cols = d + 1  # unknown coefficient columns (incl. intercept)
    pivot_row = 0
    for c in range(cols):
        # find a pivot in column c at/under pivot_row
        sel = None
        for r in range(pivot_row, n):
            if M[r][c] != 0:
                sel = r
                break
        if sel is None:
            continue
        M[pivot_row], M[sel] = M[sel], M[pivot_row]
        pv = M[pivot_row][c]
        M[pivot_row] = [v / pv for v in M[pivot_row]]
        for r in range(n):
            if r != pivot_row and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[pivot_row])]
        pivot_row += 1
        if pivot_row == n:
            break
    # Inconsistent iff some row has all-zero coeff columns but nonzero rhs.
    for r in range(n):
        if all(M[r][c] == 0 for c in range(cols)) and M[r][cols] != 0:
            return False
    return True


def analyze_kernel(header_text, blocks):
    hdr = parse_header(header_text)
    grid = hdr.get("grid", (1, 1, 1))
    gx, gy = grid[0], grid[1]

    mode_counts = {0: 0, 1: 0, 2: 0}
    n_inst = 0
    n_mem = 0
    cf_seqs = {}                 # pc-sequence tuple -> count of warps
    cf_seq_by_block = {}         # block coords -> set of seq hashes
    perpc = {}                   # pc -> list of (block_lin, warp, occ, base, mode, stride)

    for coords, warp, insts in iter_warps(blocks):
        cx, cy, cz = coords
        seq = []
        occ_by_pc = {}
        for line in insts:
            rec = parse_inst(line)
            if rec is None:
                continue
            n_inst += 1
            seq.append(rec["pc"])
            if rec["is_mem"]:
                n_mem += 1
                mode_counts[rec["mode"]] += 1
                occ = occ_by_pc.get(rec["pc"], 0)
                occ_by_pc[rec["pc"]] = occ + 1
                if rec["mode"] == 1:
                    sig = ("s", rec["stride"])
                elif rec["mode"] == 2:
                    sig = ("d", tuple(rec.get("payload", ())))
                else:
                    sig = None  # list_all: unstructured / data-dependent
                perpc.setdefault(rec["pc"], []).append(
                    (cx, cy, cz, warp, occ, rec["base"], rec["mode"], sig))
        h = hash(tuple(seq))
        cf_seqs[h] = cf_seqs.get(h, 0) + 1
        cf_seq_by_block.setdefault(coords, set()).add(h)

    # CF regularity
    n_warps = sum(cf_seqs.values())
    n_distinct_cf = len(cf_seqs)
    # divergence-by-block-coords only? (each block executes one seq)
    block_uniform = all(len(s) == 1 for s in cf_seq_by_block.values())

    # Per static mem PC: synthesizable iff structured (stride/delta), the
    # intra-warp offset signature is block-invariant, and the base address is an
    # exact affine function of (block_lin, warp, occ).
    synth_pcs = 0
    total_pcs = 0
    synth_weighted = 0
    mem_weighted = 0
    n_sig_classes = []
    for pc, recs in perpc.items():
        total_pcs += 1
        cnt = len(recs)
        mem_weighted += cnt
        # rec = (cx, cy, cz, warp, occ, base, mode, sig)
        # (1) signature must be index-determined: each (cx,cy,cz,warp) sees one sig.
        bw_sig = {}
        sig_ok = True
        for r in recs:
            key = (r[0], r[1], r[2], r[3])
            if key in bw_sig and bw_sig[key] != r[7]:
                sig_ok = False
                break
            bw_sig[key] = r[7]
        # (2) per signature class, base is exactly affine in (cx,cy,cz,warp,occ).
        is_synth = False
        if sig_ok:
            groups = {}
            for r in recs:
                groups.setdefault(r[7], []).append(r)
            n_sig_classes.append(len(groups))
            is_synth = True
            for sig, grp in groups.items():
                if sig is None:           # list_all: data-dependent
                    is_synth = False
                    break
                sample = grp if len(grp) <= 3000 else grp[::max(1, len(grp) // 3000)]
                X = [(r[0], r[1], r[2], r[3], r[4]) for r in sample]
                y = [r[5] for r in sample]
                if not affine_exact(X, y):
                    is_synth = False
                    break
        if is_synth:
            synth_pcs += 1
            synth_weighted += cnt

    return {
        "name": hdr.get("name", "?"),
        "grid": grid, "block": hdr.get("block"),
        "n_inst": n_inst, "n_mem": n_mem,
        "mode_counts": mode_counts,
        "n_warps": n_warps, "n_distinct_cf": n_distinct_cf,
        "block_uniform": block_uniform,
        "total_mem_pcs": total_pcs, "synth_pcs": synth_pcs,
        "synth_weighted": synth_weighted, "mem_weighted": mem_weighted,
    }


def verdict(k):
    mc = k["mode_counts"]
    tot = max(1, sum(mc.values()))
    listall = mc[0] / tot
    syn = k["synth_weighted"] / max(1, k["mem_weighted"])
    cf_regular = k["n_distinct_cf"] <= 1
    if k["n_mem"] == 0:
        return "TIER0/1 compute-only (no global mem)"
    if syn >= 0.99 and cf_regular:
        return "TIER1 affine — fully synthesizable"
    if syn >= 0.99 and k["block_uniform"]:
        return "TIER2 piecewise-affine — synthesizable (block-keyed CF)"
    if syn >= 0.80:
        return "TIER1/2 mostly synthesizable + small recorded remainder"
    if syn >= 0.40:
        return "TIER5 indirect — partial synth, record gather remainder"
    return "TIER4/6 data-dependent — PRNG (seed) or record"


def main():
    dirs = sys.argv[1:]
    print(f"{'kernel':36} {'grid':>14} {'mem%':>6} "
          f"{'strd/delt/list':>16} {'synth%':>6} {'cfseq':>6} verdict")
    print("-" * 130)
    for d in dirs:
        klist = glob.glob(os.path.join(d, "kernel-*.trace.zst"))
        klist.sort()
        label = os.path.basename(os.path.dirname(os.path.dirname(d))) or d
        # analyze only the first kernel file per dir to keep P0 fast
        for kf in klist[:1]:
            header, blocks = decode_trace_zst(kf)
            k = analyze_kernel(header, blocks)
            mc = k["mode_counts"]; tot = max(1, sum(mc.values()))
            mempct = 100.0 * k["n_mem"] / max(1, k["n_inst"])
            modestr = f"{100*mc[1]//tot}/{100*mc[2]//tot}/{100*mc[0]//tot}"
            synw = 100.0 * k["synth_weighted"] / max(1, k["mem_weighted"])
            name = (label + ":" + k["name"])[:36]
            print(f"{name:36} {str(k['grid']):>14} {mempct:5.1f} "
                  f"{modestr:>16} {synw:5.0f} {k['n_distinct_cf']:6d} {verdict(k)}")


if __name__ == "__main__":
    main()
