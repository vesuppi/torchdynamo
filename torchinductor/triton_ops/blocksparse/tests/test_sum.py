import sys
import torch
import triton 
from torchinductor.triton_ops.blocksparse.utils import *
from torchinductor.triton_ops.blocksparse.sum import sum as kernel

VERBOSE = False

def bench_triton(a): 
    times = []
    for (BM, BN) in [(16, 16), (32, 32), (64, 64)]:
        a_data, a_mask = to_block_format_with_mask_bmm_one_mask(a, BM, BN)
        #print(a_data[0,0])
        a_mask = RaggedFormat.from_dense_mask(a_mask)
        b_data = kernel(a_mask, a_data)
        b_ref = torch.sum(a, axis=-1)
        #print(b_ref.shape, b_data.shape)
        assert torch.allclose(b_ref, b_data), (torch.squeeze(b_ref), torch.squeeze(b_data))
        
        try:
            ms0, _, _ = triton.testing.do_bench(lambda: kernel(a_mask, a_data), rep=50)
        except Exception as e:
            print(e)
        else:
            times.append((ms0, BM, BN))
            if VERBOSE:
                print((ms0, BM, BN))
    times.sort(key=lambda x: x[0])
    #print(times)
    return times[0][0]


def bench_kernel(a, config=''):
    B, M, N = a.shape
    ms0, _, _ = triton.testing.do_bench(lambda: torch.sum(a, axis=-1))
    if VERBOSE:
        print(f'torch: {ms0:.4f}')
    ms1 = bench_triton(a)
    print(config, f'{B}x{M}x{N}', f'{ms0:.4f}', f'{ms1:.4f}', sep='; ')


def test_shape(shape, configs):
    dtype = torch.float32 
    B, M, N = shape
    a = torch.ones([B, M, N], dtype=dtype, device='cuda')
    if 'dense' in configs:
        bench_kernel(a, 'dense')
    if 'tril' in configs:
        a = torch.tril(a)
        bench_kernel(a, 'tril')


def test_seqlen_128_to_4K(configs, batch_size=96):
    for seqlen in [128, 256, 512, 1024, 2048]:
    #for seqlen in [4096]:
        test_shape((batch_size, seqlen, seqlen), configs)


if '-v' in sys.argv:
    VERBOSE = True


configs = []
if '--dense' in sys.argv:
    test_seqlen_128_to_4K(['dense'])
if '--tril' in sys.argv:
    test_seqlen_128_to_4K(['tril'])

