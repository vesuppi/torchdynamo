import sys
import torch
import triton 
import triton.language as tl
from torchinductor.triton_ops.blocksparse.utils import *


@triton.jit
def _kernel_ragged(a_mask_rowptrs, a_cols, a_vals, b_vals, c_vals, 
            M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
            BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, 
            nBM: tl.constexpr, nBK: tl.constexpr, nBN: tl.constexpr,
            ):
    m = tl.program_id(0)
    n = tl.program_id(1)
    bid = tl.program_id(2)

    # pid = tl.program_id(0)
    # bid = tl.program_id(1)
    # m = pid // nBN
    # n = pid % nBN

    a_block_size = BM * BK
    b_block_size = BK * BN
    a_ptrs = a_vals + a_block_size * nBK * m + \
        tl.arange(0, BM)[:, None] * BK + tl.arange(0, BK)[None, :]
    b_ptrs = b_vals + b_block_size * n + \
        tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]

    # b_cols = n * BN + tl.arange(0, BN)
    # b_ptrs = b_vals + tl.arange(0, BK)[:, None] * N + b_cols[None, :]


    a_ptrs += bid * M * K
    b_ptrs += bid * K * N

    #a_cols = tl.multiple_of(a_cols, 8)

    k_start = tl.load(a_cols + 2*m)
    k_end = tl.load(a_cols + 2*m+1)
    a_ptrs = a_ptrs+a_block_size * k_start
    b_ptrs = b_ptrs+b_block_size * nBN * k_start

    c = tl.zeros((BM, BN), dtype=tl.float32)

    # for k in range(nBK):
    #     a = tl.load(a_ptrs)
    #     b = tl.load(b_ptrs)
    #     c += tl.dot(a, b)
    #     a_ptrs += a_block_size
    #     b_ptrs += b_block_size * nBN

    for _ in range(k_start, k_end):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c += tl.dot(a, b)
        a_ptrs += a_block_size
        b_ptrs += b_block_size * nBN

    c = c.to(tl.float16)

    c_ptrs = c_vals + (m * nBN + n) * BM * BN + \
        tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]

    c_ptrs += bid * M * N
    
    tl.store(c_ptrs, c)


def bmm_ragged(B, M, K, N, BM, BK, BN, a_mask_rowptrs, a_cols, a_vals, b_vals, c, num_warps=4, num_stages=3):
    nBM = cdiv(M, BM)
    nBN = cdiv(N, BN)
    nBK = cdiv(K, BK)
    grid = (nBM, nBN, B)
    binary = _kernel_ragged[grid](a_mask_rowptrs, a_cols, a_vals, b_vals, c,
                            M, K, N, 
                            BM, BK, BN, nBM, nBK, nBN, 
                            num_warps=num_warps, num_stages=num_stages
                            )
    #print(binary.asm['ptx'])
    return c


## This kernel is to test the impact of `tl.swizzle2d`, which doesn't seem to make a difference
@triton.jit
def _kernel_with_swizzle2d(a_cols, a_vals, b_vals, c_vals, 
            M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
            BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, 
            nBM: tl.constexpr, nBK: tl.constexpr, nBN: tl.constexpr,
            GROUP_SIZE_M: tl.constexpr
            ):
    m = tl.program_id(0)
    n = tl.program_id(1)
    bid = tl.program_id(2)

    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    n, m = tl.swizzle2d(n, m, num_pid_n, num_pid_m, GROUP_SIZE_M)

    # pid = tl.program_id(0)
    # bid = tl.program_id(1)
    # m = pid // nBN
    # n = pid % nBN

    a_block_size = BM * BK
    b_block_size = BK * BN
    a_ptrs = a_vals + a_block_size * nBK * m + \
        tl.arange(0, BM)[:, None] * BK + tl.arange(0, BK)[None, :]
    b_ptrs = b_vals + b_block_size * n + \
        tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]

    # b_cols = n * BN + tl.arange(0, BN)
    # b_ptrs = b_vals + tl.arange(0, BK)[:, None] * N + b_cols[None, :]


    a_ptrs += bid * M * K
    b_ptrs += bid * K * N

    #a_cols = tl.multiple_of(a_cols, 8)

    k_start = tl.load(a_cols + 2*m)
    k_end = tl.load(a_cols + 2*m+1)
    a_ptrs = a_ptrs+a_block_size * k_start
    b_ptrs = b_ptrs+b_block_size * nBN * k_start

    c = tl.zeros((BM, BN), dtype=tl.float32)

    for _ in range(k_start, k_end):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c += tl.dot(a, b)
        a_ptrs += a_block_size
        b_ptrs += b_block_size * nBN

    c = c.to(tl.float16)

    c_ptrs = c_vals + (m * nBN + n) * BM * BN + \
        tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]

    c_ptrs += bid * M * N
    
    tl.store(c_ptrs, c)


def bmm_with_swizzle2d(B, M, K, N, BM, BK, BN, a_cols, a_vals, b_vals, c, num_warps=4, num_stages=3):
    nBM = cdiv(M, BM)
    nBN = cdiv(N, BN)
    nBK = cdiv(K, BK)
    grid = (nBM, nBN, B)
    binary = _kernel_with_swizzle2d[grid](a_cols, a_vals, b_vals, c,
                            M, K, N, 
                            BM, BK, BN, nBM, nBK, nBN, GROUP_SIZE_M=4,
                            num_warps=num_warps, num_stages=num_stages
                            )
    #print(binary.asm['ptx'])
    return c



## This is to test the effect of reuse by calculating two blocks per thread.
## Wasn't able to create a working implementation, currently buggy and don't know why
@triton.jit
def _kernel_2blocks(a_cols, a_vals, b_vals, c_vals, 
            M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
            BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, 
            nBM: tl.constexpr, nBK: tl.constexpr, nBN: tl.constexpr,
            ):
    m = tl.program_id(0)
    n = tl.program_id(1)
    bid = tl.program_id(2)

    n0 = n
    n1 = n * 2 + 1

    # pid = tl.program_id(0)
    # bid = tl.program_id(1)
    # m = pid // nBN
    # n = pid % nBN

    a_block_size = BM * BK
    b_block_size = BK * BN
    a_ptrs = a_vals + a_block_size * nBK * m + \
        tl.arange(0, BM)[:, None] * BK + tl.arange(0, BK)[None, :]
    b0_ptrs = b_vals + b_block_size * n0 + \
        tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]

    # b_cols = n * BN + tl.arange(0, BN)
    # b_ptrs = b_vals + tl.arange(0, BK)[:, None] * N + b_cols[None, :]


    a_ptrs += bid * M * K
    b0_ptrs += bid * K * N

    #a_cols = tl.multiple_of(a_cols, 8)

    k_start = tl.load(a_cols + 2*m)
    k_end = tl.load(a_cols + 2*m+1)
    a_ptrs = a_ptrs+a_block_size * k_start
    b0_ptrs = b0_ptrs + b_block_size * nBN * k_start
    

    c0 = tl.zeros((BM, BN), dtype=tl.float32)
    c1 = tl.zeros((BM, BN), dtype=tl.float32)

    for _ in range(k_start, k_end):
        a = tl.load(a_ptrs)
        b0 = tl.load(b0_ptrs)
        c0 += tl.dot(a, b0)
        b1_ptrs = b0_ptrs + b_block_size
        b1 = tl.load(b1_ptrs)
        c1 += tl.dot(a, b1)
        a_ptrs += a_block_size
        b0_ptrs += b_block_size * nBN
        

    c0 = c0.to(tl.float16)
    c1 = c1.to(tl.float16)

    #c0 = c0 + c1

    c0_ptrs = c_vals + (m * nBN + n0) * BM * BN + \
        tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    # c1_ptrs = c_vals + (m * nBN + n1) * BM * BN + \
    #     tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]

    c0_ptrs += bid * M * N
    # c1_ptrs += bid * M * N
    
    tl.store(c0_ptrs, c0)
    #c1_ptrs = c0_ptrs + BM * BN
    #tl.store(c1_ptrs, c0)


def bmm_2blocks(B, M, K, N, BM, BK, BN, a_cols, a_vals, b_vals, c, num_warps=4, num_stages=3):
    nBM = cdiv(M, BM)
    nBN = cdiv(N, BN)
    nBK = cdiv(K, BK)
    grid = (nBM, nBN, B)
    #print('grid:', grid)
    binary = _kernel_2blocks[grid](a_cols, a_vals, b_vals, c,
                            M, K, N, 
                            BM, BK, BN, nBM, nBK, nBN, 
                            num_warps=num_warps, num_stages=num_stages
                            )
    #print(binary.asm['ptx'])
    return c


## This is to test the effect of having a CSR-ragged format. 
## Results show that the use of two level loop seems to have a perf hit.
@triton.jit
def _kernel_ragged_outer_csr(a_mask_rowptrs, a_cols, a_vals, b_vals, c_vals, 
            M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
            BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, 
            nBM: tl.constexpr, nBK: tl.constexpr, nBN: tl.constexpr,
            ):
    m = tl.program_id(0)
    n = tl.program_id(1)
    bid = tl.program_id(2)

    # pid = tl.program_id(0)
    # bid = tl.program_id(1)
    # m = pid // nBN
    # n = pid % nBN

    a_block_size = BM * BK
    b_block_size = BK * BN
    a_ptrs = a_vals + a_block_size * nBK * m + \
        tl.arange(0, BM)[:, None] * BK + tl.arange(0, BK)[None, :]
    b_ptrs = b_vals + b_block_size * n + \
        tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]

    # b_cols = n * BN + tl.arange(0, BN)
    # b_ptrs = b_vals + tl.arange(0, BK)[:, None] * N + b_cols[None, :]


    a_ptrs += bid * M * K
    b_ptrs += bid * K * N

    #a_cols = tl.multiple_of(a_cols, 8)

    row_start = tl.load(a_mask_rowptrs+m)
    row_end = tl.load(a_mask_rowptrs+m+1)

    c = tl.zeros((BM, BN), dtype=tl.float32)

    for i in range(row_start, row_end, 2):
        k_start = tl.load(a_cols + i)
        k_end = tl.load(a_cols + i + 1)
        a_ptrs = a_ptrs+a_block_size * k_start
        b_ptrs = b_ptrs+b_block_size * nBN * k_start

        for _ in range(k_start, k_end):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            c += tl.dot(a, b)
            a_ptrs += a_block_size
            b_ptrs += b_block_size * nBN


    c = c.to(tl.float16)

    c_ptrs = c_vals + (m * nBN + n) * BM * BN + \
        tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]

    c_ptrs += bid * M * N
    
    tl.store(c_ptrs, c)


def bmm_ragged_outer_csr(B, M, K, N, BM, BK, BN, a_mask_rowptrs, a_cols, a_vals, b_vals, c, num_warps=4, num_stages=3):
    nBM = cdiv(M, BM)
    nBN = cdiv(N, BN)
    nBK = cdiv(K, BK)
    grid = (nBM, nBN, B)
    binary = _kernel_ragged_outer_csr[grid](a_mask_rowptrs, a_cols, a_vals, b_vals, c,
                            M, K, N, 
                            BM, BK, BN, nBM, nBK, nBN, 
                            num_warps=num_warps, num_stages=num_stages
                            )
    #print(binary.asm['ptx'])
    return c



## This is to test a new format, which is more general than just ragged, which uses a slow
## path and a fast path. The fast path is ragged and the slow path is CSR (to be completed)
@triton.jit
def _kernel_hybrid(a_mask_rowptrs, a_cols, a_vals, b_vals, c_vals, 
            M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
            BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, 
            nBM: tl.constexpr, nBK: tl.constexpr, nBN: tl.constexpr,
            ):
    m = tl.program_id(0)
    n = tl.program_id(1)
    bid = tl.program_id(2)

    a_block_size = BM * BK
    b_block_size = BK * BN
    
    a_ptrs = a_vals + a_block_size * nBK * m + \
        tl.arange(0, BM)[:, None] * BK + tl.arange(0, BK)[None, :]
    b_ptrs = b_vals + b_block_size * n + \
        tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]

    # b_cols = n * BN + tl.arange(0, BN)
    # b_ptrs = b_vals + tl.arange(0, BK)[:, None] * N + b_cols[None, :]


    a_ptrs += bid * M * K
    b_ptrs += bid * K * N

    #a_cols = tl.multiple_of(a_cols, 8)

    k_start = tl.load(a_cols + 2*m)
    k_end = tl.load(a_cols + 2*m+1)
    a_ptrs = a_ptrs+a_block_size * k_start
    b_ptrs = b_ptrs+b_block_size * nBN * k_start

    c = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(k_end, k_start, -1):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c += tl.dot(a, b)
        a_ptrs += a_block_size
        b_ptrs += b_block_size * nBN
    c = c.to(tl.float16)

    nnz = tl.load(a_mask_rowptrs+m)
    if nnz > 1:
        # To handle other nonzeros
        pass




    c_ptrs = c_vals + (m * nBN + n) * BM * BN + \
        tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]

    c_ptrs += bid * M * N
    
    tl.store(c_ptrs, c)


def bmm(B, M, K, N, BM, BK, BN, a_mask_rowptrs, a_cols, a_vals, b_vals, c, num_warps=4, num_stages=3):
    nBM = cdiv(M, BM)
    nBN = cdiv(N, BN)
    nBK = cdiv(K, BK)
    grid = (nBM, nBN, B)
    binary = _kernel_hybrid[grid](a_mask_rowptrs, a_cols, a_vals, b_vals, c,
                            M, K, N, 
                            BM, BK, BN, nBM, nBK, nBN, 
                            num_warps=num_warps, num_stages=num_stages
                            )
    #print(binary.asm['ptx'])
    return c
