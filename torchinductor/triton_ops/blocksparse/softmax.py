import triton
import triton.language as tl
from torchinductor.triton_ops.blocksparse.utils import *
from torchinductor.triton_ops.blocksparse.sum import sum
from torchinductor.triton_ops.blocksparse.exp import exp
from torchinductor.triton_ops.blocksparse.div import div


def softmax_unfused(x_mask, x_data):
    mask, t0 = exp(x_mask, x_data)
    t1 = sum(mask, t0, axis=1)
    mask, t2 = div(mask, t0, t1)
    return (mask, t2)


def num_warps(n):
    if n <= 128:
        return 1
    if n <= 256:
        return 2
    if n <= 512:
        return 4
    if n <= 4096:
        return 8
    return 16


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=2, num_warps=2),
        triton.Config({}, num_stages=3, num_warps=2),
        triton.Config({}, num_stages=4, num_warps=2),
        
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=4),

        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N']
)
@triton.jit
def _softmax_kernel(x_rowptrs, x_cols, x_data, y_data, 
                M: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr,
                TM: tl.constexpr, TN: tl.constexpr, 
                use_dense_data: tl.constexpr
                ):
    inner_id = tl.program_id(0)
    outer_id = tl.program_id(1)
    bid = tl.program_id(2)
    
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
    x_offsets = x_data + offsets + bid * M * N
    y_offsets = y_data + offsets + bid * M * N

    sum = tl.zeros([TM], dtype=tl.float32)
    for _ in range(col_start, col_end):
        block = tl.load(x_offsets)
        block = tl.exp(block)
        sum += tl.sum(block, axis=1)
        x_offsets += block_size

    x_offsets = x_data + offsets + bid * M * N
    for _ in range(col_start, col_end):
        # A minor optimization to save one store by redundant computation
        block = tl.load(x_offsets) 
        block = tl.exp(block)
        y = block / sum[:, None]
        
        tl.store(y_offsets, y)

        x_offsets += block_size
        y_offsets += block_size
    

def softmax(x_mask: RaggedFormat, x_data, axis=1):
    '''
    Launch a 1D grid to do the computation (blocking rows only).
    '''
    if axis != 1:
        raise Exception('Only axis=1 is supported')
    
    B, m, n, BM, BN = x_data.shape
    M = m * BM
    N = n * BN
    y_data = torch.empty_like(x_data)
    TM = 16  # Tunable parameter
    grid = (BM//TM, m, B)
    _softmax_kernel[grid](
        x_mask.rowptrs, x_mask.cols, x_data, y_data,
        M, N, BM, BN, TM, BN, True,
        #num_warps=num_warps(N)
    )
    
    y_mask = x_mask.copy()
    y_mask.default = 0
    return (y_mask, y_data)

