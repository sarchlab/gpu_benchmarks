package ptrace

import (
	"encoding/binary"
	"encoding/json"
	"strings"
	"testing"

	"github.com/klauspost/compress/zstd"
)

func containerFixture(t testing.TB) (zstdTraceHeader, []byte, []byte) {
	t.Helper()
	data, err := EncodeContainer(sampleTraceg())
	if err != nil {
		t.Fatal(err)
	}
	n := int(binary.LittleEndian.Uint32(data[:4]))
	var h zstdTraceHeader
	if err := json.Unmarshal(data[4:4+n], &h); err != nil {
		t.Fatal(err)
	}
	return h, data[4+n:], data
}

func packContainerHeader(t testing.TB, h zstdTraceHeader, body []byte) []byte {
	t.Helper()
	j, err := json.Marshal(h)
	if err != nil {
		t.Fatal(err)
	}
	return packContainerJSON(j, body)
}

func packContainerJSON(j, body []byte) []byte {
	data := make([]byte, 4, 4+len(j)+len(body))
	binary.LittleEndian.PutUint32(data, uint32(len(j)))
	data = append(data, j...)
	return append(data, body...)
}

func requireContainerError(t *testing.T, data []byte) {
	t.Helper()
	h, blocks, err := DecodeContainer(data)
	if err == nil {
		t.Fatal("malformed container accepted")
	}
	if h != "" || blocks != nil {
		t.Fatal("failed decode returned partial output")
	}
}

func TestContainerRejectsMalformedLayout(t *testing.T) {
	original, body, _ := containerFixture(t)
	maxInt := int(^uint(0) >> 1)
	cases := []struct {
		name string
		edit func(*zstdTraceHeader)
	}{
		{"missing version", func(h *zstdTraceHeader) { h.Version = 0 }},
		{"future version", func(h *zstdTraceHeader) { h.Version = 2 }},
		{"negative header", func(h *zstdTraceHeader) { h.HeaderSize = -1 }},
		{"header integer overflow", func(h *zstdTraceHeader) { h.HeaderSize = maxInt }},
		{"negative dictionary", func(h *zstdTraceHeader) { h.DictSize = -1 }},
		{"dictionary integer overflow", func(h *zstdTraceHeader) { h.DictSize = maxInt }},
		{"zero blocks", func(h *zstdTraceHeader) { h.NumBlocks = 0 }},
		{"empty table", func(h *zstdTraceHeader) {
			h.NumBlocks, h.BlockOffsets, h.FrameSizes = 0, nil, nil
		}},
		{"negative blocks", func(h *zstdTraceHeader) { h.NumBlocks = -1 }},
		{"huge block allocation", func(h *zstdTraceHeader) { h.NumBlocks = maxInt }},
		{"too few blocks", func(h *zstdTraceHeader) { h.NumBlocks-- }},
		{"missing offsets", func(h *zstdTraceHeader) { h.BlockOffsets = nil }},
		{"short offsets", func(h *zstdTraceHeader) { h.BlockOffsets = h.BlockOffsets[:1] }},
		{"extra offsets", func(h *zstdTraceHeader) { h.BlockOffsets = append(h.BlockOffsets, 0) }},
		{"missing sizes", func(h *zstdTraceHeader) { h.FrameSizes = nil }},
		{"short sizes", func(h *zstdTraceHeader) { h.FrameSizes = h.FrameSizes[:1] }},
		{"extra sizes", func(h *zstdTraceHeader) { h.FrameSizes = append(h.FrameSizes, 1) }},
		{"negative first offset", func(h *zstdTraceHeader) { h.BlockOffsets[0] = -1 }},
		{"initial gap", func(h *zstdTraceHeader) { h.BlockOffsets[0] = 1 }},
		{"frame gap", func(h *zstdTraceHeader) { h.BlockOffsets[1]++ }},
		{"frame overlap", func(h *zstdTraceHeader) { h.BlockOffsets[1]-- }},
		{"repeated frame", func(h *zstdTraceHeader) { h.BlockOffsets[1] = 0 }},
		{"negative second offset", func(h *zstdTraceHeader) { h.BlockOffsets[1] = -1 }},
		{"offset integer overflow", func(h *zstdTraceHeader) { h.BlockOffsets[1] = maxInt }},
		{"zero first frame", func(h *zstdTraceHeader) { h.FrameSizes[0] = 0 }},
		{"negative first frame", func(h *zstdTraceHeader) { h.FrameSizes[0] = -1 }},
		{"first frame integer overflow", func(h *zstdTraceHeader) { h.FrameSizes[0] = maxInt }},
		{"zero final frame", func(h *zstdTraceHeader) { h.FrameSizes[1] = 0 }},
		{"negative final frame", func(h *zstdTraceHeader) { h.FrameSizes[1] = -1 }},
		{"final frame integer overflow", func(h *zstdTraceHeader) { h.FrameSizes[1] = maxInt }},
		{"frame exceeds file", func(h *zstdTraceHeader) { h.FrameSizes[1]++ }},
		{"unindexed final byte", func(h *zstdTraceHeader) { h.FrameSizes[1]-- }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := original
			h.BlockOffsets = append([]int(nil), original.BlockOffsets...)
			h.FrameSizes = append([]int(nil), original.FrameSizes...)
			tc.edit(&h)
			requireContainerError(t, packContainerHeader(t, h, body))
		})
	}
}

func TestContainerRejectsEveryTruncatedPrefix(t *testing.T) {
	_, _, data := containerFixture(t)
	// Keep the original backing capacity: a decoder that checks capacity or
	// relies only on slice bounds can read bytes beyond the supplied prefix.
	for cut := 0; cut < len(data); cut++ {
		if _, _, err := DecodeContainer(data[:cut]); err == nil {
			t.Fatalf("accepted truncated prefix %d/%d", cut, len(data))
		}
	}
	requireContainerError(t, append([]byte(nil), data[:len(data)-1]...))
	requireContainerError(t, append(data, 0))
	t.Logf("rejected all %d proper prefixes and appended trailing data", len(data))
}

func TestContainerRejectsInvalidJSONAndFrames(t *testing.T) {
	h, body, data := containerFixture(t)
	for _, j := range []string{"", "null", "{}", "[]", "{", `{"version":1.5}`, `{"version":1,"num_blocks":9223372036854775808}`} {
		t.Run("json_"+j, func(t *testing.T) {
			requireContainerError(t, packContainerJSON([]byte(j), body))
		})
	}
	oversizedJSON := append([]byte(nil), data...)
	binary.LittleEndian.PutUint32(oversizedJSON[:4], ^uint32(0))
	requireContainerError(t, oversizedJSON)
	for i, offset := range h.BlockOffsets {
		corrupt := append([]byte(nil), body...)
		corrupt[h.HeaderSize+h.DictSize+offset] ^= 0xff
		t.Run("corrupt_frame_"+string(rune('0'+i)), func(t *testing.T) {
			requireContainerError(t, packContainerHeader(t, h, corrupt))
		})
	}
}

func dictionaryContainer(t testing.TB, id uint32, checksum bool) []byte {
	t.Helper()
	header, blocks := splitTracegBlocks(sampleTraceg())
	plain, err := zstd.NewWriter(nil, zstd.WithEncoderCRC(checksum))
	if err != nil {
		t.Fatal(err)
	}
	defer plain.Close()
	dict, err := zstd.NewWriter(nil, zstd.WithEncoderDictRaw(id, []byte(blocks[0])), zstd.WithEncoderCRC(checksum))
	if err != nil {
		t.Fatal(err)
	}
	defer dict.Close()
	h := zstdTraceHeader{Version: 1, NumBlocks: len(blocks), HeaderSize: len(header), DictSize: len(blocks[0])}
	body := []byte(header + blocks[0])
	offset := 0
	for i, block := range blocks {
		encoder := dict
		if i == 0 {
			encoder = plain
		}
		frame := encoder.EncodeAll([]byte(block), nil)
		h.BlockOffsets = append(h.BlockOffsets, offset)
		h.FrameSizes = append(h.FrameSizes, len(frame))
		offset += len(frame)
		body = append(body, frame...)
	}
	return packContainerHeader(t, h, body)
}

func TestContainerDictionaryIDCompatibility(t *testing.T) {
	// C++ uses raw dictionary ID 0 without checksums; Go uses ID 1 with them.
	// Exercise a second block so dictionary decoding is required.
	for _, id := range []uint32{0, 1} {
		gotHeader, gotBlocks, err := DecodeContainer(dictionaryContainer(t, id, id == 1))
		if err != nil {
			t.Fatalf("dictionary ID %d: %v", id, err)
		}
		if gotHeader+strings.Join(gotBlocks, "") != sampleTraceg() {
			t.Fatalf("dictionary ID %d changed decoded bytes", id)
		}
	}
}

func TestContainerRejectsDictionaryMismatch(t *testing.T) {
	for _, id := range []uint32{0, 1} {
		data := dictionaryContainer(t, id, id == 1)
		n := int(binary.LittleEndian.Uint32(data[:4]))
		var h zstdTraceHeader
		if err := json.Unmarshal(data[4:4+n], &h); err != nil {
			t.Fatal(err)
		}
		for _, at := range []int{0, h.DictSize / 2, h.DictSize - 1} {
			corrupt := append([]byte(nil), data...)
			corrupt[4+n+h.HeaderSize+at] ^= 1
			requireContainerError(t, corrupt)
		}
		// Even a one-block file must match: no later frame otherwise uses the
		// dictionary, so successful decompression alone cannot detect damage.
		body := append([]byte(nil), data[4+n:4+n+h.HeaderSize+h.DictSize+h.FrameSizes[0]]...)
		h.NumBlocks, h.BlockOffsets, h.FrameSizes = 1, h.BlockOffsets[:1], h.FrameSizes[:1]
		if _, _, err := DecodeContainer(packContainerHeader(t, h, body)); err != nil {
			t.Fatal(err)
		}
		body[h.HeaderSize] ^= 1
		requireContainerError(t, packContainerHeader(t, h, body))
	}
}

func FuzzContainerMetadata(f *testing.F) {
	h, body, _ := containerFixture(f)
	j, err := json.Marshal(h)
	if err != nil {
		f.Fatal(err)
	}
	f.Add(j)
	f.Add([]byte(`{"version":1,"num_blocks":-1}`))
	f.Add([]byte("null"))
	f.Fuzz(func(t *testing.T, metadata []byte) {
		// Keep the compressed payload fixed and small. This targets metadata
		// arithmetic/indexing rather than decompression capacity policy.
		if len(metadata) > 8192 {
			t.Skip()
		}
		DecodeContainer(packContainerJSON(metadata, body))
	})
}
