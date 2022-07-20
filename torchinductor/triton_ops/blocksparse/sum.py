import sys
import torch
import triton 
import triton.language as tl
from torchinductor.triton_ops.blocksparse.utils import *


@triton.jit
def _sum_kernel(x_rowptrs, x_cols, x_data, y_data, 
                M: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr,
                TM: tl.constexpr, TN: tl.constexpr, 
                use_dense_data: tl.constexpr
                ):
    inner_id = tl.program_id(0)
    outer_id = tl.program_id(1)
    
    block_size = BM * BN

    ## Format specific: how to get `k` would depend on the format
    col_start = tl.load(x_cols + 2*outer_id)
    col_end = tl.load(x_cols + 2*outer_id+1)

    offsets = 0
    ## If data layout is dense - good for debugging
    if use_dense_data:
        offsets += outer_id * BM * N 
    else:
        # TODO: add indexing for sparse data blocks
        pass

    offsets += inner_id * TM * BN + tl.arange(0, TM)[:, None] * BN + tl.arange(0, BN)[None, :]
    x_offsets = x_data + offsets

    sum = tl.zeros([TM], dtype=tl.float32)
    for _ in range(col_start, col_end):
        block = tl.load(x_offsets)
        sum += tl.sum(block, axis=1)
        x_offsets += block_size    

    y_offsets = y_data + outer_id * BM + inner_id * TM + tl.arange(0, TM)
    tl.store(y_offsets, sum)
    

def sum(x_mask: RaggedFormat, x_data, axis=1):
    '''
    Launch a 1D grid to do the computation (blocking rows only).
    '''
    if axis != 1:
        raise Exception('Only axis=1 is supported')
    
    B, m, n, BM, BN = x_data.shape
    M = m * BM
    N = n * BN
    y_data = torch.empty([B, M], dtype=x_data.dtype, device='cuda')
    TM = 8  # Tunable parameter
    grid = (BM//TM, m, B)
    _sum_kernel[grid](
        x_mask.rowptrs, x_mask.cols, x_data, y_data,
        M, N, BM, BN, TM, BN, True
    )
    
    return y_data

