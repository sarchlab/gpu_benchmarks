import json
import os
import sys
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float32)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def benchmark():
    device = 'cuda'

    # Read parameters from environment variables
    m = int(os.environ.get('BENCH_PARAM_m', '4096'))
    block_m = int(os.environ.get('BENCH_PARAM_block_m', '128'))
    block_n = int(os.environ.get('BENCH_PARAM_block_n', '256'))
    block_k = int(os.environ.get('BENCH_PARAM_block_k', '64'))
    group_size_m = int(os.environ.get('BENCH_PARAM_group_size_m', '8'))
    num_warmup = 5

    # Square matrices: M = N = K
    M_dim, N_dim, K_dim = m, m, m

    print(f"Running triton_gemm with m={m}, block_m={block_m}, block_n={block_n}, block_k={block_k}, group_size_m={group_size_m}", file=sys.stderr)

    a = torch.rand((M_dim, K_dim), device=device, dtype=torch.float32)
    b = torch.rand((K_dim, N_dim), device=device, dtype=torch.float32)
    c = torch.empty((M_dim, N_dim), device=device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M_dim, meta['BLOCK_M']) * triton.cdiv(N_dim, meta['BLOCK_N']),)

    # Warmup
    for _ in range(num_warmup):
        matmul_kernel[grid](
            a, b, c,
            M_dim, N_dim, K_dim,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP_SIZE_M=group_size_m,
        )
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        matmul_kernel[grid](
            a, b, c,
            M_dim, N_dim, K_dim,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP_SIZE_M=group_size_m,
        )
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / 5

    tflops = 2 * M_dim * N_dim * K_dim / (avg_ms / 1000) / 1e12

    params = {
        "m": m,
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "group_size_m": group_size_m,
    }

    print(json.dumps({
        "type": "kernel",
        "name": "matmul_kernel",
        "time_ms": avg_ms,
        "params": params,
    }))

    print(json.dumps({
        "type": "summary",
        "total_time_ms": avg_ms,
        "metrics": [
            {"name": "tflops", "value": tflops},
        ],
    }))

if __name__ == '__main__':
    benchmark()
