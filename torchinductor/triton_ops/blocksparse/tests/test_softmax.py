import sys
import torch
import triton 
from torchinductor.triton_ops.blocksparse.utils import *
from torchinductor.triton_ops.blocksparse.softmax import softmax as kernel
from triton.ops.blocksparse.softmax import softmax as triton_softmax

VERBOSE = False

def mysoftmax(a, axis=-1):
    #return a
    return torch.softmax(a, axis=axis)
    e = torch.exp(a)
    s = torch.sum(e, axis=axis)
    return e / s[:, :, None]


def bench_triton(a): 
    times = []
    for (BM, BN) in [(16, 16), (32, 32), (64, 64)]:
    #for (BM, BN) in [(32, 32)]:
        b_ref = mysoftmax(a, axis=-1)
        a_data, a_mask = to_sparseblock_with_dense_mask(a, BM, BN, compressed_val=-torch.inf)
        
        func = triton_softmax(a_mask, BM, 'cuda')
        b_mask = RaggedFormat.from_dense_mask(a_mask.squeeze(), default=0)
        b_data = func(a_data)

        b_data_ref, _ = to_sparseblock_with_dense_mask(b_ref, BM, BN, compressed_val=0)
        #print(b_data.shape, b_data_ref.shape)
        assert torch.allclose(b_data, b_data_ref)
        #print(b_data)
        #print(b_ref)

        
        #if torch.all(a_mask == True):
        #    assert torch.allclose(b_ref, b_data), (b_ref, b_data)
        
        ms0, _, _ = triton.testing.do_bench(lambda: func(a_data), rep=50)
        times.append((ms0, BM, BN))
    #print(times)
    times.sort(key=lambda x: x[0])
    return times[0][0]


def bench_ours(a): 
    times = []
    for (BM, BN) in [(16, 16), (32, 32), (64, 64)]:
        a_blocked = FastBCSR.to_format(a, BM, BN, is_blocks_dense=True, compressed_val=-torch.inf)
        b_blocked = kernel(a_blocked)
        b_ref = mysoftmax(a, axis=-1)
        b_ref_blocked, _ = to_block_format_with_mask_bmm_one_mask(b_ref, BM, BN)
        
        assert torch.allclose(b_ref_blocked, b_blocked.vals), (b_ref_blocked, b_blocked.vals)
        try:
            ms0, _, _ = triton.testing.do_bench(lambda: kernel(a_blocked), rep=50)
        except Exception as e:
            print(e)
        else:
            times.append((ms0, BM, BN))
            if VERBOSE:
                print((ms0, BM, BN))
    times.sort(key=lambda x: x[0])
    return times[0][0]


def bench_ours_new(a): 
    times = []
    for (BM, BN) in [(16, 16), (32, 32), (64, 64)]:
        a_sparse = FastBCSR.to_format(a, BM, BN, compressed_val=-torch.inf)
        b_sparse = kernel(a_sparse)
        b_ref = mysoftmax(a, axis=-1)
        b_data_ref, _ = to_sparseblock_with_dense_mask(b_ref, BM, BN, compressed_val=0)
        #print(b_data_ref.shape)
        assert torch.allclose(b_data_ref, b_sparse.vals, atol=0.25, rtol=0.05), (b_data_ref[0,-1,0], b_sparse.vals[0,-1,0])
        try:
            ms0, _, _ = triton.testing.do_bench(lambda: kernel(a_sparse), rep=50)
        except Exception as e:
            print(e)
        else:
            times.append((ms0, BM, BN))
            if VERBOSE:
                print((ms0, BM, BN))
    times.sort(key=lambda x: x[0])
    return times[0][0]


def bench_kernel(a, config=''):
    B, M, N = a.shape
    ms0, _, _ = triton.testing.do_bench(lambda: mysoftmax(a, axis=-1))
    if VERBOSE:
        print(f'torch: {ms0:.4f}')
    ms1 = bench_triton(a)
    ms2 = bench_ours_new(a)
    print(config, f'{B}x{M}x{N}', f'{ms0:.4f}', f'{ms1:.4f}', f'{ms2:.4f}', sep='; ')


def test_shape(shape, configs):
    dtype = torch.float32 
    B, M, N = shape
    a = torch.ones([B, M, N], dtype=dtype, device='cuda')
    if 'dense' in configs:
        bench_kernel(a, 'dense')
    if 'tril' in configs:
        a = torch.tril(a)
        a = a.masked_fill_(a == 0, -torch.inf)
        bench_kernel(a, 'tril')


def test_seqlen_128_to_4K(configs, batch_size=96):
    for seqlen in [128, 256, 512, 1024, 2048]:
    #for seqlen in [4096]:
        test_shape((batch_size, seqlen, seqlen), configs)
        #break


if '-v' in sys.argv:
    VERBOSE = True


configs = []
if '--dense' in sys.argv:
    test_seqlen_128_to_4K(['dense'])
if '--tril' in sys.argv:
    test_seqlen_128_to_4K(['tril'])