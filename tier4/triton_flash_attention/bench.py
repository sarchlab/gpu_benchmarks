import torch
import triton
import triton.language as tl
import math

@triton.jit
def flash_attention_fwd(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    H, S,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H

    # Offset base pointers for this batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    O += off_b * stride_ob + off_h * stride_oh

    # Block offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    # Load Q block: (BLOCK_M, BLOCK_D)
    q = tl.load(Q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)

    # Initialize online softmax accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Iterate over K/V blocks
    for start_n in range(0, S, BLOCK_N):
        cur_n = start_n + offs_n

        # Load K^T: (BLOCK_D, BLOCK_N) — transposed for Q @ K^T
        k = tl.load(K + offs_d[:, None] * stride_kd + cur_n[None, :] * stride_ks)

        # Compute attention scores: S = Q @ K^T * scale
        qk = tl.dot(q, k) * sm_scale

        # Online softmax: update running max and sum
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Load V block: (BLOCK_N, BLOCK_D)
        v = tl.load(V + cur_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)

        # Update output accumulator: rescale old + add new
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output
    tl.store(O + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
             acc.to(tl.float16))

def benchmark():
    device = 'cuda'
    B, H, S, D = 2, 16, 1024, 64
    BLOCK_M, BLOCK_N, BLOCK_D = 128, 64, 64
    sm_scale = 1.0 / math.sqrt(D)

    Q = torch.randn((B, H, S, D), device=device, dtype=torch.float16)
    K = torch.randn((B, H, S, D), device=device, dtype=torch.float16)
    V = torch.randn((B, H, S, D), device=device, dtype=torch.float16)
    O = torch.empty_like(Q)

    grid = (triton.cdiv(S, BLOCK_M), B * H)

    # Warmup
    for _ in range(5):
        flash_attention_fwd[grid](
            Q, K, V, O,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            H, S, sm_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        flash_attention_fwd[grid](
            Q, K, V, O,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            H, S, sm_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 5

    tflops = 4 * B * H * S * S * D / (ms / 1000) / 1e12
    print(f'triton_flash_attention,{B}x{H}x{S}x{D},{ms:.3f},{tflops:.2f}')

if __name__ == '__main__':
    benchmark()
