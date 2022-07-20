import sys
import torch
import triton 
from torchinductor.triton_ops.blocksparse.utils import *
from torchinductor.triton_ops.blocksparse.div import div_colvec as kernel

VERBOSE = False

def bench_triton_exp(a, b): 
    times = []
    for BM in [16, 32, 64]:
        for BN in [16, 32, 64]:
            a_data, a_mask = to_block_format_with_mask_bmm_one_mask(a, BM, BN)
            a_mask = RaggedFormat.from_dense_mask(a_mask)
            c_mask, c_data = kernel(a_mask, a_data, b)
            c_dense = c_mask.to_dense(c_data)

            c_ref = a/(b[:, :, None])
            assert torch.allclose(c_ref, c_dense), (c_ref, c_dense)
            for num_warps in [2,4]:
                for num_stages in [2,3,4]:
                    try:
                        ms0, _, _ = triton.testing.do_bench(lambda: kernel(a_mask, a_data, b), rep=50)
                    except Exception as e:
                        print(e)
                    else:
                        times.append((ms0, BM, BN, num_warps, num_stages))
                        if VERBOSE:
                            print((ms0, BM, BN, num_warps, num_stages))
    times.sort(key=lambda x: x[0])
    return times[0][0]


def bench_kernel(a, b, config=''):
    B, M, N = a.shape
    ms0, _, _ = triton.testing.do_bench(lambda: a/(b[:, :, None]))
    ms1 = bench_triton_exp(a, b)
    print(config, f'{B}x{M}x{N}', f'{ms0:.4f}', f'{ms1:.4f}', sep='; ')


def test_configs(configs):
    dtype = torch.float32 
    for B in [1]:
        for M in [1024, 2048, 4096]:
            for N in [1024, 2048, 4096]:
                a = torch.rand([B, M, N], dtype=dtype, device='cuda')
                b = torch.rand([B, M], dtype=dtype, device='cuda')
                if 'dense' in configs:
                    bench_kernel(a, b, 'dense')
                if 'tril' in configs:
                    a = torch.tril(a)
                    bench_kernel(a, b, 'tril')


if '-v' in sys.argv:
    VERBOSE = True


if '--tril' in sys.argv:
    test_configs(['tril'])

if '--dense' in sys.argv:
    test_configs(['dense'])


