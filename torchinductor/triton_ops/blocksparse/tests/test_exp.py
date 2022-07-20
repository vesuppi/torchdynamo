import sys
import torch
import triton 
from torchinductor.triton_ops.blocksparse.utils import *
from torchinductor.triton_ops.blocksparse.exp import kernel

VERBOSE = False

def bench_triton_exp(a): 
    times = []
    for BM in [16, 32, 64]:
        for BN in [16, 32, 64]:
            a_data, a_mask = to_block_format_with_mask_bmm_one_mask(a, BM, BN)
            a_mask = RaggedFormat.from_dense_mask(a_mask)
            b_mask, b_data = kernel(a_mask, a_data)
            b_dense = b_mask.to_dense(b_data)
            assert torch.allclose(torch.exp(a), b_dense), \
                (torch.exp(a)[0,0], b_dense[0,0])
            for num_warps in [2,4,8]:
                for num_stages in  [2,3,4]:
                    try:
                        ms0, _, _ = triton.testing.do_bench(lambda: kernel(a_mask, a_data), rep=50)
                    except Exception as e:
                        print(e)
                    else:
                        times.append((ms0, BM, BN, num_warps, num_stages))
                        if VERBOSE:
                            print((ms0, BM, BN, num_warps, num_stages))
    times.sort(key=lambda x: x[0])
    return times[0][0]


def bench_kernel(a, config=''):
    B, M, N = a.shape
    ms0, _, _ = triton.testing.do_bench(lambda: torch.exp(a))
    ms1 = bench_triton_exp(a)
    print(config, f'{B}x{M}x{N}', f'{ms0:.4f}', f'{ms1:.4f}', sep='; ')


def test_configs(configs):
    dtype = torch.float32 
    for B in [1]:
        for M in [1024, 2048, 4096]:
            for N in [1024, 2048, 4096]:
                a = torch.rand([B, M, N], dtype=dtype, device='cuda')
                if 'dense' in configs:
                    bench_kernel(a, 'dense')
                if 'tril' in configs:
                    a = torch.tril(a)
                    a = a.masked_fill_(a == 0, -torch.inf)
                    bench_kernel(a, 'tril')


if '-v' in sys.argv:
    VERBOSE = True


configs = []
if '--tril' in sys.argv:
    configs.append('tril')
if '--dense' in sys.argv:
    configs.append('dense')

test_configs(configs)
