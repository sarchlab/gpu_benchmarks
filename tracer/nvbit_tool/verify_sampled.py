#!/usr/bin/env python3
"""Verify a sampled .trace.zst against a full .trace.zst of the same run config.

Every block present in the sampled trace must match the same-coordinate block
in the full trace token-for-token, except tokens that are hex addresses
(0x...), which are run-specific (cudaMalloc bases differ between processes).
Mirrors the addr-shaped/structural classification in tracer/ptrace/regime.go.
"""
import json
import re
import sys

import zstandard


def load_blocks(path):
    """Return {(x,y,z): block_text} from a v1 .trace.zst container."""
    with open(path, "rb") as f:
        raw = f.read()
    jlen = int.from_bytes(raw[:4], "little")
    hdr = json.loads(raw[4 : 4 + jlen])
    off = 4 + jlen + hdr["header_size"]
    dict_bytes = raw[off : off + hdr["dict_size"]]
    frames_base = off + hdr["dict_size"]

    blocks = {}
    for i in range(hdr["num_blocks"]):
        fo = frames_base + hdr["block_offsets"][i]
        frame = raw[fo : fo + hdr["frame_sizes"][i]]
        if i == 0:
            text = zstandard.ZstdDecompressor().decompress(frame)
        else:
            d = zstandard.ZstdDecompressor(
                dict_data=zstandard.ZstdCompressionDict(dict_bytes)
            )
            text = d.decompress(frame)
        text = text.decode()
        m = re.search(r"thread block = (\d+),(\d+),(\d+)", text)
        blocks[tuple(map(int, m.groups()))] = text
    return blocks


HEX = re.compile(r"^0x[0-9a-fA-F]+$")

full = load_blocks(sys.argv[1])
samp = load_blocks(sys.argv[2])

total_addr_diffs = 0
total_struct_diffs = 0
for coord, stext in sorted(samp.items()):
    ftext = full.get(coord)
    if ftext is None:
        print(f"block {coord}: MISSING from full trace")
        total_struct_diffs += 1
        continue
    if stext == ftext:
        print(f"block {coord}: byte-identical")
        continue
    stok, ftok = stext.split(), ftext.split()
    if len(stok) != len(ftok):
        print(f"block {coord}: STRUCTURAL — token count {len(stok)} vs {len(ftok)}")
        total_struct_diffs += 1
        continue
    addr_d = struct_d = 0
    for a, b in zip(ftok, stok):
        if a == b:
            continue
        if HEX.match(a) and HEX.match(b):
            addr_d += 1
        else:
            struct_d += 1
    total_addr_diffs += addr_d
    total_struct_diffs += struct_d
    tag = "OK (addr-only)" if struct_d == 0 else "STRUCTURAL DIFFS"
    print(f"block {coord}: {tag} — {addr_d} address diffs, {struct_d} structural")

print(
    f"\nsampled blocks: {len(samp)}  full blocks: {len(full)}  "
    f"addr diffs: {total_addr_diffs}  structural diffs: {total_struct_diffs}"
)
sys.exit(1 if total_struct_diffs else 0)
