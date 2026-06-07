#!/usr/bin/env python3
"""Decode bb_triosim .trace.zst container -> plaintext .traceg.

Container format (from bb_triosim/internal/codec/zstd.go):
  [4-byte LE hdr-json length][hdr JSON][traceg header bytes][dict bytes][frames...]
Each thread block is an independent zstd frame. Block 0 is compressed with no
dictionary; its raw text is then the raw-content dictionary for blocks 1..N.
"""
import json
import struct
import sys

import zstandard as zstd


def decode_trace_zst(path):
    with open(path, "rb") as f:
        data = f.read()
    (hdr_len,) = struct.unpack_from("<I", data, 0)
    off = 4
    hdr = json.loads(data[off:off + hdr_len])
    off += hdr_len
    traceg_header = data[off:off + hdr["header_size"]].decode("utf-8", "replace")
    off += hdr["header_size"]
    dict_bytes = data[off:off + hdr["dict_size"]]
    off += hdr["dict_size"]
    frames_base = off

    offsets = hdr["block_offsets"]
    sizes = hdr["frame_sizes"]

    # Frame 0: no dict. Its decompressed output is the raw dictionary.
    d_plain = zstd.ZstdDecompressor()
    f0 = data[frames_base + offsets[0]: frames_base + offsets[0] + sizes[0]]
    block0 = d_plain.decompress(f0)
    # NOTE: reads C++ NVBit-tracer output (raw-dict frames, dictID 0). The Go
    # `ptrace` tool is canonical for reading .ptrace / Go-written containers.
    raw_dict = zstd.ZstdCompressionDict(block0, dict_type=zstd.DICT_TYPE_RAWCONTENT)
    d_dict = zstd.ZstdDecompressor(dict_data=raw_dict)

    blocks = [block0]
    for i in range(1, len(offsets)):
        fr = data[frames_base + offsets[i]: frames_base + offsets[i] + sizes[i]]
        blocks.append(d_dict.decompress(fr))
    return traceg_header, blocks


def main():
    header, blocks = decode_trace_zst(sys.argv[1])
    sys.stdout.write(header)
    for b in blocks:
        sys.stdout.write(b.decode("utf-8", "replace"))


if __name__ == "__main__":
    main()
