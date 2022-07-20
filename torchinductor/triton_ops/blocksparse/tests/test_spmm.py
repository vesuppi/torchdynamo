import sys
import argparse
import torch
import triton 
import triton.language as tl
from torchinductor.triton_ops.blocksparse.utils import *
from torchinductor.triton_ops.blocksparse.mmconfigs import basic_configs
from torchinductor.triton_ops.blocksparse.spmm import bmm5 as kernel
from torchinductor.triton_ops.batched_matmul import bmm_out

    
def test_lower_triangular(B, M, K, N, is_tril=True, runtime_log=sys.stdout):
    print(f'{B}x{M}x{K}x{N}')
    dtype = torch.float16
    a = torch.randn([B, M, K], dtype=dtype, device='cuda')
    if is_tril:
        a = torch.tril(a)
    b = torch.randn([B, K, N], dtype=dtype, device='cuda')
    c_ref = torch.empty([B, M, N], dtype=dtype, device='cuda')
    torch_ms, _, _ = triton.testing.do_bench(lambda: torch.bmm(a, b, out=c_ref))
    print(f'info: torch bmm: {torch_ms:.4f}')

    triton_c_ref = torch.empty([B, M, N], dtype=dtype, device='cuda')
    triton_ms = 0
    triton_ms, _, _ = triton.testing.do_bench(lambda: bmm_out(a, b, triton_c_ref))
    
    print(f'info: triton bmm: {triton_ms:.4f}')
    print(torch.allclose(c_ref, triton_c_ref, atol=0.1, rtol=0.01))

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
        a_mask = RaggedFormat.from_dense_mask(a_mask)
        a_mask_rowptrs, a_mask_cols = a_mask.rowptrs, a_mask.cols
        
        #print(a_mask_rowptrs)
        
        b_block, b_mask = to_block_format_with_mask_bmm_one_mask(b, BK, BN)
        c = gen_empty_matrix_dense_blocks(M, N, BM, BN, batch_size=B)

        ms = torch.inf
        try:
            ms, _, _ = triton.testing.do_bench(lambda: 
                kernel(B, M, K, N, BM, BK, BN, a_mask_rowptrs, a_mask_cols, a_block, b_block, c[1], num_warps, num_stages), 
            rep=50)
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
    print(f'{B}x{M}x{K}x{N}', f'{torch_ms:.4f}', f'{triton_ms:.4f}', f'{best_time:.4f}', sep='; ', file=runtime_log)
    runtime_log.flush()
    

def test_post_shapes_lower_tri(test_dense=False):
    runtime_log = open(f'results/post.log', 'w')
    print(f'# Test post shapes, also test dense case: {test_dense}', file=runtime_log)
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
        test_lower_triangular(B, M, K, N, runtime_log=runtime_log)

    if test_dense:
        for shape in shapes:
            B, M, K, N = shape
            test_lower_triangular(B, M, K, N, False, runtime_log=runtime_log)
    runtime_log.close()


def test_single_batch():
    shapes = [
        (1, 1024, 1024, 1024),
        (1, 1024, 1024, 4096),
        (1, 1024, 1024, 8192),
        (1, 2048, 2048, 1024//16),
        (1, 2048, 2048, 4096//16),
        (1, 2048, 2048, 8192//16),
        (1, 4096, 4096, 4096),
        (1, 8192, 8192, 8192),
    ]
    

def test_torchbench_shapes(test_dense=False):
    runtime_log = open(f'results/torchbench.log', 'w')
    print(f'# Test torchbench shapes, also test dense case: {test_dense}', file=runtime_log)
    shapes = [
        (192, 128, 64, 128),
        (192, 128, 128, 64),
        (12, 1024, 1024, 64),
        (12, 1024, 64, 1024),
        (12, 512, 64, 512),
        (12, 512, 512, 64),
    ]
    for shape in shapes:
        B, M, K, N = shape
        test_lower_triangular(B, M, K, N, runtime_log=runtime_log)

    if test_dense:
        for shape in shapes:
            B, M, K, N = shape
            test_lower_triangular(B, M, K, N, False, runtime_log=runtime_log)
    runtime_log.close()



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-m', type=int, default=0)
parser.add_argument('-k', type=int)
parser.add_argument('-n', type=int)
parser.add_argument('-b', type=int)
parser.add_argument('-t', type=str, default="")
args = parser.parse_args()

B, M, K, N = args.b, args.m, args.k, args.n

if args.t == 'post':
    test_post_shapes_lower_tri()
elif args.t == 'torchbench':
    test_torchbench_shapes(True)
else:
    test_lower_triangular(B, M, K, N)

