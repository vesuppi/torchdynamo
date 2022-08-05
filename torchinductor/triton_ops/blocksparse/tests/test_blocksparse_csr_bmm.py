import sys
import argparse
import torch
import triton 
import triton.language as tl
from torchinductor.triton_ops.blocksparse.utils import *
from triton.ops.blocksparse import matmul as blocksparse_matmul
from torchinductor.triton_ops.batched_matmul import bmm_out
from torchinductor.triton_ops.blocksparse.utils import *
from torchinductor.triton_ops.blocksparse.mmconfigs import basic_configs

@triton.jit
def _kernel_mcsr_bmm(a_rowptrs, a_cols, a_vals, b_vals, c_vals, 
                                BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, 
                                nBM: tl.constexpr, nBK: tl.constexpr, nBN: tl.constexpr,
                                ):
    m = tl.program_id(0)
    n = tl.program_id(1)
    bid = tl.program_id(2)

    M: tl.constexpr = BM * nBM
    K: tl.constexpr = BK * nBK
    N: tl.constexpr = BN * nBN

    a_block_size = BM * BK
    b_block_size = BK * BN
    a_ptrs = a_vals + a_block_size * nBK * m + \
        tl.arange(0, BM)[:, None] * BK + tl.arange(0, BK)[None, :]
    b_ptrs = b_vals + b_block_size * n + \
        tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]

    a_ptrs += bid * M * K
    b_ptrs += bid * K * N

    k_start = tl.load(a_rowptrs+m)
    k_end = tl.load(a_rowptrs+m+1)
    c = tl.zeros((BM, BN), dtype=tl.float32)
    
    a_cols_ptr = a_cols + k_start

    for kp in range(k_start, k_end):
        k = tl.load(a_cols_ptr)
        a = tl.load(a_ptrs+a_block_size*k)
        b = tl.load(b_ptrs+b_block_size * nBN*k)
        c += tl.dot(a, b)

        a_cols_ptr += 1

    c = c.to(tl.float16)

    c_ptrs = c_vals + (m * nBN + n) * BM * BN + \
        tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]

    c_ptrs += bid * M * N
    
    tl.store(c_ptrs, c)


def mcsr_bmm_inner(a_rowptrs, a_cols, a_vals, b_vals, c, num_warps=4, num_stages=3):
    B, nBM, nBK, BM, BK = a_vals.shape
    B, nBK, nBN, BK, BN = b_vals.shape
    # TODO: this does not work when M does not divide BM
    # Or maybe it works because C will also need to be padded
    M = nBM * BM 
    N = nBN * BN

    grid = (nBM, nBN, B)
    binary = _kernel_mcsr_bmm[grid](a_rowptrs, a_cols, a_vals, b_vals, c,
                                    BM, BK, BN, nBM, nBK, nBN, 
                                    num_warps=num_warps, num_stages=num_stages
                                    )
    #print(binary.asm['ptx'])
    return c

    
def test_lower_triangular(B, M, K, N, runtime_log=sys.stdout):
    dtype = torch.float16
    a = torch.randn([B, M, K], dtype=dtype, device='cuda')
    a = torch.tril(a)
    b = torch.randn([B, K, N], dtype=dtype, device='cuda')
    c_ref = torch.empty([B, M, N], dtype=dtype, device='cuda')
    torch_ms, _, _ = triton.testing.do_bench(lambda: torch.matmul(a, b, out=c_ref))
    print(f'info: torch bmm: {torch_ms:.4f}')
    triton_ms = 0

    times = []
    for config in basic_configs:
        BM = config.kwargs['BLOCK_M']
        BN = config.kwargs['BLOCK_N']
        BK = config.kwargs['BLOCK_K']
  
        if BM > M or BK > K or BN > N:
            continue

        num_stages = config.num_stages
        num_warps = config.num_warps
        print(f'info: blocks: {BM} x {BK} x {BN}')
        a_block, a_mask = to_block_format_with_mask_bmm_one_mask(a, BM, BK)
        a_mask_rowptrs, a_mask_cols = to_csr_ptrs(a_mask)
        b_block, b_mask = to_block_format_with_mask_bmm_one_mask(b, BK, BN)
        #print(a_mask_rowptrs, a_mask_cols)
        c = gen_empty_matrix_dense_blocks(M, N, BM, BN, batch_size=B)
        
        #print(a_mask_cols)
        
        b_block, b_mask = to_block_format_with_mask_bmm_one_mask(b, BK, BN)
        #print(a_mask_rowptrs, a_mask_cols)
        c = gen_empty_matrix_dense_blocks(M, N, BM, BN, batch_size=B)

        ms = torch.inf
        try:
            ms, _, _ = triton.testing.do_bench(lambda: mcsr_bmm_inner(a_mask_rowptrs, a_mask_cols, a_block, b_block, c[1], num_warps, num_stages), rep=50)
            print(f'info: {num_stages} x {num_warps}, {ms:.4f}')
            
        except Exception as e:
            print('info: run triton failed ({BM} x {BK} x {BN})')
            print(type(e))
            print(e)
        verified = torch.allclose(c_ref, from_block_format(c[1]))
        print('info: verify passes:', verified)
        if verified:
            times.append((ms, BM, BK, BN, num_stages, num_warps))
    times.sort(key=lambda x: x[0])
    best_time = times[0][0]
    #print(f'info: blocksparse mm: {times[0][0]:.4f} ({BM} x {BK} x {BN})')
    print(f'{B}x{M}x{K}x{N}', f'{torch_ms:.4f}', f'{triton_ms:.4f}', f'{best_time:.4f}', sep='; ', file=runtime_log)
    runtime_log.flush()
    

def test_post_shapes_lower_tri():
    shapes = [
        (32*16, 1024, 1024, 1024//16),
        (32*16, 1024, 1024, 4096//16),
        (32*16, 1024, 1024, 8192//16),
        (32*16, 2048, 2048, 1024//16),
        (32*16, 2048, 2048, 4096//16),
        (32*16, 2048, 2048, 8192//16),
    ]
    for shape in shapes:
        B, M, K, N = shape
        test_lower_triangular(B, M, K, N)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-m', type=int, default=0)
parser.add_argument('-k', type=int)
parser.add_argument('-n', type=int)
parser.add_argument('-b', type=int)
args = parser.parse_args()

B, M, K, N = args.b, args.m, args.k, args.n

if M == 0:
    test_post_shapes_lower_tri()
else:
    test_lower_triangular(B, M, K, N)