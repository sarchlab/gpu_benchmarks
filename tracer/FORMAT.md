# `.ptrace` Format Specification (v1)

`.ptrace` is gpubench's trace container. v1 (P1) is a **lossless re-encoding** of
the Accel-Sim trace the NVBit tracer emits: it hoists the per-PC-invariant fields
into a shared static table and stores only the varying data per dynamic
instruction. It does **no address modeling yet** (that is P2); the address
payload is stored verbatim.

`ptrace expand` reconstructs the original Accel-Sim `.traceg` content
byte-for-byte, so BBSim consumes expanded output unchanged.

## Directory layout

```
<name>.ptrace/
  manifest.json        ordered kernelslist entries, contexts, invocations
  isa/<hash>.json      static per-PC table, deduped across invocations
  dyn/inv_<n>.zst      columnar dynamic payload (zstd), one per invocation
```

## The Accel-Sim line, decomposed

An instruction line (lineinfo off, no core-id) is space-separated with a trailing
space:

```
PC  mask  dst_num [Rdst]  opcode  src_num [Rsrc...]  W [mode payload...]  imm
```

- **Static (per PC):** `dst_num [Rdst]`, `opcode`, `src_num [Rsrc...]`, the
  memory width `W` (or the non-memory `0` marker), `is_mem`, and `imm`.
  `imm` is the encoded immediate / branch target and is invariant per PC.
- **Dynamic (per instruction):** `mask`, and for memory ops the address
  compression `mode` ∈ {0 `list_all`, 1 `base_stride`, 2 `base_delta`} plus its
  `payload` tokens.

`convert` validates that each PC's static content is identical across every
dynamic occurrence; an inconsistency is a hard error (it would indicate the
decomposition is unsafe for that kernel).

Raw text parsing requires complete `#BEGIN_TB` / `#END_TB` frames with three
nonnegative block coordinates and no repeated block coordinates. Warp IDs must
start at zero and follow their array positions. Each warp declares one
nonnegative `insts` count that must equal its actual instruction lines. Empty
warps and blocks with no warps remain representable; sampled block coordinates
may be sparse and need not be sorted. Header text is retained verbatim.

Instruction operand counts are bounded by the available tokens before indexing.
PCs must be nonnegative and fit the host `int`; masks must fit 32 bits. Memory
widths are positive integers, and modes 0/1/2 require the corresponding complete
address payload with valid numeric fields. The non-memory `0` marker is followed
only by the immediate token. Zero-mask stride/base payloads and valid numeric
spellings are preserved, including signed strides and deltas. Malformed framing,
counts and operands return errors rather than panicking or silently renumbering
warps, truncating masks or accepting missing instructions.

These grammar checks leave opcode/register/immediate meaning and launch-header
consistency to the consuming tool. They do not establish measured-region
membership or make an empty trace sufficient collection evidence.

## `isa/<hash>.json` — static table

```json
{ "pcs":   [0, 16, 32, ...],
  "insts": [ {"m":"1 R1 LDC 0", "mem":false, "i":"0"},
             {"m":"1 R2 LDG.E 1 R4", "mem":true, "w":"4", "i":"0"} ] }
```

`pcs[i]` (virtual PC) has static content `insts[i]`. `m` is the space-joined
middle tokens (dst+opcode+srcs), `mem`/`w`/`i` as above. The table is identified
by a content hash and **stored once per distinct static program**, so a kernel
relaunched N times (iterative solvers, auto-tuned loops) shares one table.

## `dyn/inv_<n>.zst` — columnar dynamic payload (v2)

zstd (full-window) of an uncompressed buffer that starts with a 1-byte version
(`2`) followed by six length-prefixed sections:

| section | contents |
|---------|----------|
| **A — structure** | `uvarint nBlocks`; per block: `uvarint x,y,z`, `uvarint nWarps`; per warp: `uvarint nInsts` |
| **B — pc column**   | per instruction (block→warp→inst order): `uvarint pcId` (index into `isa.pcs`) |
| **C — mask column** | per instruction: 4-byte LE `uint32` active-and-predicate mask |
| **R — recipes**     | `uvarint nModeledPCs`; per PC: `uvarint pcId`, `byte multiclass`, `uvarint nClasses`; per class: `byte mode`, `uvarint nTail` + tail tokens, 6× signed-varint affine coefficients |
| **S — selector**    | per **modeled multiclass** memory instruction: `uvarint classId` |
| **D — residual addr** | per **recorded** (non-modeled) memory instruction: `byte mode`; `uvarint nTokens`; per token `uvarint len` + bytes |

Decode is sequential: read A (fixes instruction count and block/warp shape), then
B and C, then R (building a `pcId → recipe` map). For each memory instruction, in
canonical order:

- **modeled PC** → pick the class (id 0, or read S when multiclass); compute
  `base = c0·bx + c1·by + c2·bz + c3·warp + c4·occ + c5`; payload =
  `["0x%x"(base)] ++ tail`. No bytes are consumed from D.
- **recorded PC** → read `mode` + payload tokens from D.

`occ` is the per-warp occurrence index of the PC (reset each warp), recomputed
identically on encode and decode. See "address modeling" below.

## Address modeling (recipes)

A memory PC is **modeled** (its base addresses replaced by coefficients) only
when, for each address-encoding class (a distinct `mode` + constant `tail`):

1. the class is **index-determined** — every instance at a given `(bx,by,bz,warp)`
   uses the same class (so the choice is a function of indices, not data), and
   there is real tail reuse (`nClasses ≤ nInstances/2`, or a single class); and
2. the per-instance base address is an **exact integer-affine** function of
   `(bx,by,bz,warp,occ)` (solved over rationals; integer solution required).

Fitting is **self-verified**: each class regenerates its recorded base token and
must match byte-for-byte, else the PC falls back to recorded. Data-dependent
instructions (gather/scatter, PRNG) therefore stay in section D — lossless and
honest. This covers tiers 0–3 (constant / affine / piecewise / tiled). Tiers 4
(PRNG-from-seed) and 5 (recorded index column for gathers) are not yet emitted.

Only blocks that actually executed are stored (matching the tracer, which skips
empty blocks). Within a stored block, all warp slots `0..nWarps-1` are present;
empty warps have `nInsts = 0`.

## Sampled payload (dyn v4) — P3 synthesis

`ptrace synth` writes a **sampled** dyn payload (leading version byte 4) that
stores only a subset of blocks and reconstructs the rest. It is produced from a
full trace by `ConvertSampled`:

1. **Corner sample.** Instrument the blocks whose every coordinate is one of the
   two smallest distinct values on its axis — at most `2×2×2 = 8` blocks
   regardless of launch size. Two values per varying axis is the minimum the
   affine solver needs to fix a slope.
2. **Fit** recipes (above) on the sample only.
3. **Verify-or-fallback.** For every other block, synthesize it from the sample's
   first block (the per-warp shape *template*) + recipes, and compare byte-for-byte
   to the real block. A mismatch demotes that block to *recorded*.

Sections mirror v2 but cover only the **recorded** blocks, plus a plan:
`[A struct][B pc][C mask][R recipes][D residual][F flags][G coords]`. Recorded
blocks store every memory access in **D** verbatim (no recipe regeneration — a
block is recorded precisely because the recipe did not reproduce it), and their
struct in **A** carries *no* coordinates. **F** holds the total block count, the
template's slot in the recorded list, and a bitset (1 = synth, 0 = recorded).
**G** is the per-block coordinate stream: a zig-zag `(dx,dy,dz)` delta from the
previous block, for *every* block in order.

A synthesized block therefore costs only one flag bit plus its coordinate delta —
both near-constant streams for a dense grid, which zstd reduces to almost nothing.
The payload is then dominated by the (problem-size-independent) corner sample, so
a fully-synthesizable kernel's sampled `.ptrace` is roughly **constant size
regardless of grid**, and the compression ratio grows with the trace rather than
plateauing. The result is byte-identical to the original; the recorded fraction is
the collection-wall-time proxy.

`ptrace synth --grid GxHxD` instead extrapolates the recipes to a **new** grid
(`ExtrapolateGrid`) — generating a large trace from a small one. This has no
ground truth at the new size and is therefore **unverified**; a committed
ground-truth trace must gate any production use.

## `manifest.json`

```json
{
  "version": 1,
  "source": "accelsim-convert",
  "contexts": [
    { "kernelslist": "kernelslist_ctx_0x..",
      "stats": "stats_ctx_0x..", "stats_content": "...",
      "entries": [
        {"type":"memcpy","raw":"MemcpyHtoD,0x..,128"},
        {"type":"kernel","raw":"kernel-1-ctx_0x...trace.zsim","inv":0}
      ] }
  ],
  "invocations": [
    { "trace_file":"kernel-1-ctx_0x...trace.zst",
      "header":"-kernel name = ...\n...",
      "isa":"<hash>", "dyn":"inv_0.zst" }
  ]
}
```

`contexts[].entries` mirrors the original `kernelslist` **in order** (memcpy ops
interleaved with kernel launches) so `expand` reproduces it exactly. Each kernel
entry points at an invocation, which carries the verbatim `.traceg` header (grid
/block dims, shmem, nregs, base addresses), its ISA hash, and its dyn payload.

## Container (`.trace.zst`) — for `convert` input and `expand` output

Both the NVBit C++ tracer and bb_triosim's Go codec write the same container:

```
[4-byte LE hdr-json length][hdr JSON][traceg header bytes][dict bytes][frames...]
```

Each thread block is an independent zstd frame; block 0 is compressed without a
dictionary and its raw text is the raw-content dictionary for blocks 1..N. The
C++ tracer tags frames with dictID 0, the Go encoder with dictID 1; the reader in
`container.go` registers the raw dict under both.

The container version is 1. Its block count is positive and must equal both
frame-table lengths. Header and dictionary sizes are nonnegative. Positive
frame lengths cover the complete remaining payload consecutively, starting at
offset zero, without gaps, overlaps or trailing bytes. The decoder checks these
ranges against the supplied byte length before allocating decoded blocks or
decompressing. Unsupported versions, inconsistent tables and truncated or
out-of-range sections return errors rather than panicking or accepting partial
files. Valid containers retain their original decompressed header and block
bytes. The stored dictionary must also match decoded block 0 byte for byte;
this is checked before decoding later blocks that could silently use a corrupt
dictionary when their zstd frames have no checksum. These container checks do
not establish instruction validity, measured region membership or a host-memory
budget for decompressed traces.

## Tier-7 statistical residual (lossy, `tier7.go`)

A data-dependent gather (SpMV `x[col[]]`) is near-maximal entropy and cannot be
compressed losslessly. For cache/performance simulation only the *pattern* matters,
so tier-7 stores a regenerable statistical model instead of the addresses. One
gather PC's model (`StatModel.Bytes`) is a uvarint stream:

```
[uvarint base][uvarint elem][uvarint range][uvarint seed]
[uvarint nLaneClasses]( [uvarint laneCount][uvarint nWarpsWithThatCount] )*
[uvarint nFreq]( [uvarint bucketCount] )*        # nFreq == 0 ⇒ uniform model
```

`base + index*elem` is an address; `index` is drawn i.i.d. over `[0,range)` —
uniform when `nFreq == 0`, else from the `nFreq`-bucket frequency histogram. The
per-warp active-lane counts replay the lane-class histogram. Synthesis is seeded
(`seed`) and therefore deterministic.

This residual is **lossy and opt-in**: it is *not* part of the byte-exact `convert`
/`verify` path. It is accepted only when its regenerated stream reproduces the
original's **LRU miss-rate curve and per-warp transaction count** within tolerance
(the gate in `cache.go`); otherwise the access stays in the lossless tier-5
residual (D). A tier-7 access must never be used for functional replay.

## Versioning

`manifest.version` is 2. The dyn payload carries its own leading version byte.
v2 added the recipe (R), selector (S), and residual-addr (D) sections described
above; a fully-recorded v2 payload (no modeled PCs) is equivalent to the original
P1 lossless re-encoding. The dyn payload's leading version byte selects the
decoder: **3** = full (every block stored, v2 sections), **4** = sampled (P3,
recorded subset + synthesis plan). Future tiers (PRNG-from-seed, recorded index
column) will extend section R without changing the section framing.
