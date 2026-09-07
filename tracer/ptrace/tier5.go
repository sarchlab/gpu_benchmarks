package ptrace

// tier5.go compactly stores the addresses of *recorded* (non-modeled) memory
// instructions — gathers, scatters, PRNG accesses. Instead of the tracer's text
// payload (16-hex addresses or decimal deltas, ~6–16 bytes/lane), it reconstructs
// the per-active-lane address list and stores it as base + zig-zag varint deltas
// (~1–3 bytes/lane for the bounded indices these kernels actually touch). The
// compact form is used only when it regenerates the exact original tokens;
// otherwise a literal token list preserves information the addresses omit.

import (
	"fmt"
	"math/bits"
	"strconv"
)

// payloadToAddrs reconstructs the per-active-lane address list from a recorded
// payload, given the active mask (needed to size base_stride lists).
func payloadToAddrs(mode int, payload []string, mask uint32) ([]uint64, error) {
	n := bits.OnesCount32(mask)
	if n == 0 {
		return nil, nil
	}
	switch mode {
	case 0: // list_all: payload is the addresses
		if len(payload) != n {
			return nil, fmt.Errorf("list_all: %d tokens, %d active", len(payload), n)
		}
		out := make([]uint64, n)
		for i, t := range payload {
			v, err := parseHexAddr(t)
			if err != nil {
				return nil, err
			}
			out[i] = v
		}
		return out, nil
	case 1: // base_stride: base, stride
		if len(payload) != 2 {
			return nil, fmt.Errorf("base_stride: %d tokens", len(payload))
		}
		base, err := parseHexAddr(payload[0])
		if err != nil {
			return nil, err
		}
		stride, err := strconv.ParseInt(payload[1], 10, 64)
		if err != nil {
			return nil, err
		}
		out := make([]uint64, n)
		for k := 0; k < n; k++ {
			out[k] = uint64(int64(base) + int64(k)*stride)
		}
		return out, nil
	case 2: // base_delta: base, then n-1 deltas
		if len(payload) != n {
			return nil, fmt.Errorf("base_delta: %d tokens, %d active", len(payload), n)
		}
		base, err := parseHexAddr(payload[0])
		if err != nil {
			return nil, err
		}
		out := make([]uint64, n)
		out[0] = base
		cur := int64(base)
		for k := 1; k < n; k++ {
			d, err := strconv.ParseInt(payload[k], 10, 64)
			if err != nil {
				return nil, err
			}
			cur += d
			out[k] = uint64(cur)
		}
		return out, nil
	}
	return nil, fmt.Errorf("unknown mode %d", mode)
}

// addrsToPayload emits the canonical tracer spelling for an address list.
func addrsToPayload(mode int, addrs []uint64) []string {
	if len(addrs) == 0 {
		return nil
	}
	switch mode {
	case 0:
		out := make([]string, len(addrs))
		for i, a := range addrs {
			out[i] = fmt.Sprintf("0x%016x", a)
		}
		return out
	case 1:
		stride := int64(0)
		if len(addrs) >= 2 {
			stride = int64(addrs[1]) - int64(addrs[0])
		}
		return []string{fmt.Sprintf("0x%x", addrs[0]), strconv.FormatInt(stride, 10)}
	case 2:
		out := make([]string, len(addrs))
		out[0] = fmt.Sprintf("0x%x", addrs[0])
		for k := 1; k < len(addrs); k++ {
			out[k] = strconv.FormatInt(int64(addrs[k])-int64(addrs[k-1]), 10)
		}
		return out
	}
	return nil
}

func parseHexAddr(t string) (uint64, error) {
	if len(t) >= 2 && (t[:2] == "0x" || t[:2] == "0X") {
		t = t[2:]
	}
	return strconv.ParseUint(t, 16, 64)
}

// residualLiteral distinguishes an exact token list from the compact address
// representation. Dynamic versions 5/6 introduce this tag; versions 3/4 cannot
// represent payloads that are not recoverable from the active-lane addresses.
const residualLiteral = 0xff

// encodeResidual uses base + deltas only when regenerating the tokens is exact.
// Zero active lanes, one active lane with a nonzero stride, and noncanonical
// address spelling can otherwise discard information from the recorded trace.
func encodeResidual(buf []byte, mode int, payload []string, mask uint32) ([]byte, error) {
	if mode < 0 || mode > 2 {
		return nil, fmt.Errorf("unknown mode %d", mode)
	}
	addrs, err := payloadToAddrs(mode, payload, mask)
	if err != nil {
		return nil, err
	}
	if !eqStr(addrsToPayload(mode, addrs), payload) {
		buf = append(buf, residualLiteral, byte(mode))
		buf = putUvarint(buf, uint64(len(payload)))
		for _, token := range payload {
			buf = putUvarint(buf, uint64(len(token)))
			buf = append(buf, token...)
		}
		return buf, nil
	}
	buf = append(buf, byte(mode))
	if len(addrs) == 0 {
		return buf, nil
	}
	buf = putUvarint(buf, addrs[0])
	prev := int64(addrs[0])
	for k := 1; k < len(addrs); k++ {
		buf = putVarint(buf, int64(addrs[k])-prev)
		prev = int64(addrs[k])
	}
	return buf, nil
}

// decodeResidual reads one recorded memory instruction and regenerates its
// (mode, payload) using the active mask.
func decodeResidual(c *cursor, mask uint32, allowLiteral bool) (int, []string, error) {
	mb, err := c.bytes(1)
	if err != nil {
		return 0, nil, err
	}
	mode := int(mb[0])
	if mode == residualLiteral {
		if !allowLiteral {
			return 0, nil, fmt.Errorf("literal residual in legacy dynamic format")
		}
		mb, err = c.bytes(1)
		if err != nil {
			return 0, nil, err
		}
		mode = int(mb[0])
		if mode > 2 {
			return 0, nil, fmt.Errorf("unknown literal mode %d", mode)
		}
		count, err := c.uvarint()
		if err != nil {
			return 0, nil, err
		}
		// Every token needs at least its length prefix. Bound lengths against
		// remaining bytes before allocating or converting them to int.
		if count > uint64(len(c.b)-c.pos) {
			return 0, nil, fmt.Errorf("truncated literal token list")
		}
		payload := make([]string, int(count))
		for i := range payload {
			n, err := c.uvarint()
			if err != nil {
				return 0, nil, err
			}
			if n > uint64(len(c.b)-c.pos) {
				return 0, nil, fmt.Errorf("truncated literal token")
			}
			b, err := c.bytes(int(n))
			if err != nil {
				return 0, nil, err
			}
			payload[i] = string(b)
		}
		return mode, payload, nil
	}
	if mode > 2 {
		return 0, nil, fmt.Errorf("unknown mode %d", mode)
	}
	n := bits.OnesCount32(mask)
	if n == 0 {
		return mode, nil, nil
	}
	base, err := c.uvarint()
	if err != nil {
		return 0, nil, err
	}
	addrs := make([]uint64, n)
	addrs[0] = base
	cur := int64(base)
	for k := 1; k < n; k++ {
		d, err := c.varint()
		if err != nil {
			return 0, nil, err
		}
		cur += d
		addrs[k] = uint64(cur)
	}
	return mode, addrsToPayload(mode, addrs), nil
}
