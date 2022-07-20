import sys
import torch
import triton 
import triton.language as tl
from torchinductor.triton_ops.blocksparse.utils import *


@triton.jit
def _div_kernel(x_rowptrs, x_cols, x_data, y, z_data, 
                M: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr,
                # tile sizes, BM needs to divide TM etc
                TM: tl.constexpr, TN: tl.constexpr, 
                use_dense_data: tl.constexpr
                ):
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    ## Format specific: how to get `k` would depend on the format
    col_start = tl.load(x_cols + 2*m)
    col_end = tl.load(x_cols + 2*m+1)
    ## Skip the computation if not a nonzero
    if (n >= col_end) | (n < col_start):
        return 

    block_size = BM * BN

    offsets = 0
    ## If data layout is dense - good for debugging
    if use_dense_data:
        offsets += m * BM * N 
    else:
        # TODO: add indexing for sparse data blocks
        pass
    offsets += tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :] 
    offsets += n * block_size
    x_offsets = x_data + offsets
    z_offsets = z_data + offsets
    y_offsets = y + m * BM + tl.arange(0, BM)

    ## Format specific: how to get `k` would depend on the format
    x = tl.load(x_offsets)
    y = tl.load(y_offsets)
    
    ## Kernel specific
    z = x / y[:, None]
    
    ## Format specific: how to get `k` would depend on the format
    tl.store(z_offsets, z)


def div_colvec(x_mask: RaggedFormat, x_data, y):
    '''
    `x` is a sparse matrix and `y` is a dense column vector
    '''
    B, m, n, BM, BN = x_data.shape
    M = m * BM
    N = n * BN
    z_data = torch.empty_like(x_data)
    grid = (m, n, B)    
    _div_kernel[grid](
        x_mask.rowptrs, x_mask.cols, x_data, y, z_data,
        M, N, BM, BN, BM, BN, True
    )
    # Same mask is used for y
    return (x_mask, z_data)

