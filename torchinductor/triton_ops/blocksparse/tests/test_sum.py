import sys
from sentry_sdk import configure_scope
import torch
import triton 
from torchinductor.triton_ops.blocksparse.utils import *
from torchinductor.triton_ops.blocksparse.sum import sum as kernel

VERBOSE = False

def bench_triton(a): 
    times = []
    for BM in [16, 32, 64]:
        for BN in [16, 32, 64]:
            a_data, a_mask = to_block_format_with_mask_bmm_one_mask(a, BM, BN)
            a_mask = RaggedFormat.from_dense_mask(a_mask)
            b_data = kernel(a_mask, a_data)
            b_ref = torch.sum(a, axis=-1)
            #print(b_ref.shape, b_data.shape)
            assert torch.allclose(b_ref, b_data), (b_ref, b_data)
            
            for num_warps in [2,4,8]:
                for num_stages in [3,4]:
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
    ms0, _, _ = triton.testing.do_bench(lambda: torch.sum(a, axis=-1))
    if VERBOSE:
        print(f'torch: {ms0:.4f}')
    ms1 = bench_triton(a)
    print(config, f'{B}x{M}x{N}', f'{ms0:.4f}', f'{ms1:.4f}', sep='; ')


def test_configs(configs):
    dtype = torch.float32 
    for B in [1]:
        for M in [1024, 2048, 4096]:
            for N in [1024, 2048, 4096]:
                a = torch.ones([B, M, N], dtype=dtype, device='cuda')
                if 'dense' in configs:
                    bench_kernel(a, 'dense')
                if 'tril' in configs:
                    a = torch.tril(a)
                    bench_kernel(a, 'tril')


if '-v' in sys.argv:
    VERBOSE = True


configs = []
if '--tril' in sys.argv:
    configs.append('tril')
if '--dense' in sys.argv:
    configs.append('dense')

test_configs(configs)

'''
Performance seems to vary significant for different block sizes.
'''