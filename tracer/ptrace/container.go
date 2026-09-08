package ptrace

// container.go ports the bb_triosim Accel-Sim ".trace.zst" container format so
// that `ptrace expand` produces files BBSim can read without modification, and
// `ptrace convert` can read the traces gpubench already collects.
//
// On-disk layout (little-endian):
//   [4-byte hdr-json length][hdr JSON][traceg header bytes][dict bytes][frames...]
// Each thread block is an independent zstd frame. Block 0 is compressed without a
// dictionary; its raw text is the raw-content dictionary for blocks 1..N.
// Mirrors github.com/sarchlab/bbsim/internal/codec.

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/klauspost/compress/zstd"
)

type zstdTraceHeader struct {
	Version      int   `json:"version"`
	DictSize     int   `json:"dict_size"`
	NumBlocks    int   `json:"num_blocks"`
	HeaderSize   int   `json:"header_size"`
	BlockOffsets []int `json:"block_offsets"`
	FrameSizes   []int `json:"frame_sizes"`
}

// DecodeContainer parses a .trace.zst byte blob into the .traceg header string
// and the per-thread-block text segments (in grid order).
func DecodeContainer(data []byte) (header string, blocks []string, err error) {
	if len(data) < 4 {
		return "", nil, fmt.Errorf("file too small")
	}
	hdrLen := binary.LittleEndian.Uint32(data[:4])
	pos := 4
	if uint64(hdrLen) > uint64(len(data)-pos) {
		return "", nil, fmt.Errorf("truncated at header json")
	}
	var hdr zstdTraceHeader
	if err := json.Unmarshal(data[pos:pos+int(hdrLen)], &hdr); err != nil {
		return "", nil, fmt.Errorf("parsing header json: %w", err)
	}
	pos += int(hdrLen)
	if hdr.Version != 1 {
		return "", nil, fmt.Errorf("unsupported container version %d", hdr.Version)
	}
	if hdr.NumBlocks <= 0 || len(hdr.BlockOffsets) != hdr.NumBlocks || len(hdr.FrameSizes) != hdr.NumBlocks {
		return "", nil, fmt.Errorf("invalid block table: count %d, offsets %d, sizes %d",
			hdr.NumBlocks, len(hdr.BlockOffsets), len(hdr.FrameSizes))
	}
	if hdr.HeaderSize < 0 || hdr.HeaderSize > len(data)-pos {
		return "", nil, fmt.Errorf("truncated at traceg header")
	}
	header = string(data[pos : pos+hdr.HeaderSize])
	pos += hdr.HeaderSize
	if hdr.DictSize < 0 || hdr.DictSize > len(data)-pos {
		return "", nil, fmt.Errorf("truncated at dictionary")
	}
	dict := data[pos : pos+hdr.DictSize]
	pos += hdr.DictSize
	frames := data[pos:]

	// Both version-1 writers pack positive-length frames consecutively. Check
	// the entire table against the supplied length before indexing, allocating
	// output blocks or decoding. Slice capacity may extend beyond that length,
	// and additions involving corrupt file offsets can overflow.
	end := 0
	for i, start := range hdr.BlockOffsets {
		size := hdr.FrameSizes[i]
		if start != end || size <= 0 || size > len(frames)-end {
			return "", nil, fmt.Errorf("invalid frame %d: offset %d, size %d, expected offset %d within %d bytes",
				i, start, size, end, len(frames))
		}
		end += size
	}
	if end != len(frames) {
		return "", nil, fmt.Errorf("%d unindexed trailing frame bytes", len(frames)-end)
	}

	dPlain, err := zstd.NewReader(nil)
	if err != nil {
		return "", nil, err
	}
	defer dPlain.Close()
	// C++ tracer (ZSTD_compress_usingDict, raw) tags frames with dictID 0; the Go
	// encoder uses dictID 1. Register the raw dict under both so we read either.
	dDict, err := zstd.NewReader(nil,
		zstd.WithDecoderDictRaw(0, dict),
		zstd.WithDecoderDictRaw(1, dict))
	if err != nil {
		return "", nil, err
	}
	defer dDict.Close()

	blocks = make([]string, hdr.NumBlocks)
	for i := 0; i < hdr.NumBlocks; i++ {
		start := hdr.BlockOffsets[i]
		fr := frames[start : start+hdr.FrameSizes[i]]
		var out []byte
		if i == 0 {
			out, err = dPlain.DecodeAll(fr, nil)
		} else {
			out, err = dDict.DecodeAll(fr, nil)
		}
		if err != nil {
			return "", nil, fmt.Errorf("decoding frame %d: %w", i, err)
		}
		blocks[i] = string(out)
	}
	return header, blocks, nil
}

// splitTracegBlocks splits a full .traceg string into its header and the list of
// per-thread-block segments. Each block begins at a "\n#BEGIN_TB\n" marker; the
// header is everything before the first marker. Matches the bb_triosim splitter.
func splitTracegBlocks(content string) (header string, blocks []string) {
	const marker = "\n#BEGIN_TB\n"
	idx := strings.Index(content, marker)
	if idx < 0 {
		return content, nil
	}
	header = content[:idx]
	rest := content[idx:]
	parts := strings.Split(rest, marker)
	for _, p := range parts {
		if p == "" {
			continue
		}
		blocks = append(blocks, marker+p)
	}
	return header, blocks
}

// EncodeContainer compresses a full .traceg string into the .trace.zst container
// layout BBSim reads. Inverse of DecodeContainer at the .traceg-content level.
func EncodeContainer(content string) ([]byte, error) {
	header, blocks := splitTracegBlocks(content)
	if len(blocks) == 0 {
		return nil, fmt.Errorf("no thread blocks found")
	}
	dict := []byte(blocks[0])

	encPlain, err := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedDefault))
	if err != nil {
		return nil, err
	}
	defer encPlain.Close()
	encDict, err := zstd.NewWriter(nil,
		zstd.WithEncoderLevel(zstd.SpeedDefault),
		zstd.WithEncoderDictRaw(1, dict))
	if err != nil {
		return nil, err
	}
	defer encDict.Close()

	frames := make([][]byte, len(blocks))
	offsets := make([]int, len(blocks))
	sizes := make([]int, len(blocks))
	off := 0
	for i, b := range blocks {
		var fr []byte
		if i == 0 {
			fr = encPlain.EncodeAll([]byte(b), nil)
		} else {
			fr = encDict.EncodeAll([]byte(b), nil)
		}
		frames[i] = fr
		offsets[i] = off
		sizes[i] = len(fr)
		off += len(fr)
	}

	hdr := zstdTraceHeader{
		Version:      1,
		DictSize:     len(dict),
		NumBlocks:    len(blocks),
		HeaderSize:   len(header),
		BlockOffsets: offsets,
		FrameSizes:   sizes,
	}
	hdrJSON, err := json.Marshal(hdr)
	if err != nil {
		return nil, err
	}

	var out []byte
	var lenbuf [4]byte
	binary.LittleEndian.PutUint32(lenbuf[:], uint32(len(hdrJSON)))
	out = append(out, lenbuf[:]...)
	out = append(out, hdrJSON...)
	out = append(out, header...)
	out = append(out, dict...)
	for _, fr := range frames {
		out = append(out, fr...)
	}
	return out, nil
}
