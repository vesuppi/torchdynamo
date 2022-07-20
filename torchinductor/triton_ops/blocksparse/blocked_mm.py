import sys
import torch
import triton
import triton.language as tl
import triton.testing
from triton.ops.matmul import matmul as triton_matmul
from utils import *

torch.backends.cuda.matmul.allow_tf32 = True

VERIFY = False

print_naive_PTX = False
print_block_PTX = False



@triton.jit
def _kernel_naive_mm(a_ptr, b_ptr, c_ptr, M, N, K, 
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, 
            BLOCK_SIZE_N: tl.constexpr):
    mid = tl.program_id(0)
    nid = tl.program_id(1)
    # Starting row + BLOCK_SIZE_M more rows
    a_rows = mid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Starting col + BLOCK_SIZE_N more columns
    b_cols = nid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    a_ptrs = a_ptr + a_rows[:, None] * K + tl.arange(0, BLOCK_SIZE_K)[None, :]
    b_ptrs = b_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * N + b_cols[None, :]

    c = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(K//BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

    c = c.to(tl.float16)

    # C's block's offsets
    c_ptrs = a_rows[:, None] * N + b_cols[None, :]
    tl.store(c_ptr+ c_ptrs, c)


def naive_mm(a, b, BLOCK_M, BLOCK_K, BLOCK_N, num_warps=4, num_stages=3):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty([M, N], device=a.device, dtype=a.dtype)
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    binary = _kernel_naive_mm[grid](a, b, c, M, N, K, BLOCK_M, BLOCK_K, BLOCK_N, num_stages=num_stages, num_warps=num_warps)
    global print_naive_PTX
    if print_naive_PTX:
        print('info: naive PTX')
        print(binary.asm['ptx'])
        print_naive_PTX = False
        sys.stdout.flush()
    return c



@triton.jit
def _kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    num_blocks_in_K: tl.constexpr,
    num_blocks_in_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Compute the outer_m, outer_n
    mid = tl.program_id(0)
    nid = tl.program_id(1)

    a_block_ptrs = a_ptr + mid * num_blocks_in_K * BLOCK_M * BLOCK_K + \
        tl.arange(0, BLOCK_M)[:, None] * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]

    b_block_ptrs = b_ptr + nid * BLOCK_K * BLOCK_N + \
        tl.arange(0, BLOCK_K)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    c = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(num_blocks_in_K):
        a = tl.load(a_block_ptrs)
        b = tl.load(b_block_ptrs)
        c += tl.dot(a, b)
        a_block_ptrs += BLOCK_M * BLOCK_K
        b_block_ptrs += BLOCK_K * (BLOCK_N * num_blocks_in_N)

    c = c.to(tl.float16)

    c_block_ptrs = c_ptr + (mid * num_blocks_in_N + nid) * BLOCK_M * BLOCK_N + \
        tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(c_block_ptrs, c)


def blocked_mm(a, b, num_warps=4, num_stages=3):
    outer_m_dim, outer_k_dim, BLOCK_M, BLOCK_K = a.shape
    outer_k_dim, outer_n_dim, BLOCK_K, BLOCK_N = b.shape

    M = outer_m_dim * BLOCK_M
    N = outer_n_dim * BLOCK_N
    K = outer_k_dim * BLOCK_K

    c = torch.empty(
        (outer_m_dim, outer_n_dim, BLOCK_M, BLOCK_N), device=a.device, dtype=a.dtype
    )
    grid = (outer_m_dim, outer_n_dim)
    binary = _kernel[grid](a, b, c, M, N, K, outer_k_dim, outer_n_dim, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=num_warps, num_stages=num_stages)
    global print_block_PTX
    if print_block_PTX:
        print('info: blocked PTX')
        print(binary.asm['ptx'])
        print_block_PTX = False
        sys.stdout.flush()
    return c


def check_block_format_utils():
    BLOCK = 16
    a = torch.randn(64, 64)
    blocked_a = to_block_format(a, BLOCK, BLOCK)
    dense_a = from_block_format(blocked_a)
    assert torch.allclose(a, dense_a)


def run_triton_block_mm(a, b, M, K, N, BLOCK_M, BLOCK_K, BLOCK_N):
    # Triton
    blocked_a = to_block_format(a, BLOCK_M, BLOCK_K)
    blocked_b = to_block_format(b, BLOCK_K, BLOCK_N)
    blocked_c = blocked_mm(blocked_a, blocked_b)
    res_c = from_block_format(blocked_c)
    
    ref_c = torch.mm(a, b)
    assert torch.allclose(ref_c, res_c, rtol=0.05, atol=0.1)

    times = []
    #for num_stages in [1,2,3,4,5,6]:
    for num_stages in [2,3,4,5]:
        for num_warps in [2,4,8]:
            ms, _, _ = triton.testing.do_bench(lambda: blocked_mm(blocked_a, blocked_b, num_warps, num_stages), rep=50)
            times.append((ms, num_stages, num_warps))
    times.sort(key=lambda x: x[0])
    #print('best blocked:', times[0])
    return times[0][0]


def run_triton_naive_mm(a, b, M, K, N, BLOCK_M, BLOCK_K, BLOCK_N):
    ref_c = torch.mm(a, b)
    # Triton
    
    res_c = naive_mm(a, b, BLOCK_M, BLOCK_K, BLOCK_N)
    tol = 1e-1
    assert torch.allclose(ref_c, res_c, rtol=0.05, atol=tol)

    times = []
    #for num_stages in [1,2,3,4,5,6]:
    for num_stages in [2,3,4,5]:
        for num_warps in [2,4,8]:
            ms, _, _ = triton.testing.do_bench(lambda: naive_mm(a, b, BLOCK_M, BLOCK_K, BLOCK_N, num_warps, num_stages), rep=50)
            times.append((ms, num_stages, num_warps))
    times.sort(key=lambda x: x[0])
    #print('best unblocked:', times[0])
    return times[0][0]

def run_torch(a, b, M, K, N):
    for _ in range(3):
        torch.mm(a, b)
    ms, _, _ = triton.testing.do_bench(lambda: torch.mm(a, b))
    return ms

def run_normal_triton(a, b, M, K, N):
    for _ in range(3):
        triton_matmul(a, b)
    ms, _, _ = triton.testing.do_bench(lambda: triton_matmul(a, b))
    return ms

def run_regular_shapes():
    torch.manual_seed(0)
    M = 64
    K = 128
    N = 256
    BLOCK = 32

    # dtype = torch.float16
    # for M in [2048, 4096]:
    #     for N in [2048, 4096]:
    #         for K in [2048, 4096]:
    #         #for K in [4096*2]:
    #             a = torch.randn((M, K), device="cuda", dtype=dtype)
    #             b = torch.randn((K, N), device="cuda", dtype=dtype)

    #             ms1 = run_torch(a, b, M, K, N)
    #             ms2 = run_triton(a, b, M, K, N, 64, 64, 64)
    #             FLOPS1 = 2 * M * K * N / ms1 / 10**9
    #             FLOPS2 = 2 * M * K * N / ms2 / 10**9
    #             print(M, K, N, ms1, FLOPS1, FLOPS1/312, ms2, FLOPS2, FLOPS2/312)
    # sys.exit(0)

    # for shape in [(128, 4096, 1000)]:
    #     M, K, N = shape
    #     a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    #     b = torch.randn((K, N), device="cuda", dtype=a.dtype)
    #     ms1 = run_torch(a, b, M, K, N)
    #     for blocks in [(64, 64, 64)]:
    #         BLOCK_M, BLOCK_K, BLOCK_N = blocks
    #         ms2 = run_triton(a, b, M, K, N, BLOCK_M, BLOCK_K, BLOCK_N)
    #         print(ms1, ms2)

    # return
    i = 0
    for dtype in [torch.float16, torch.float32]:
        for M in [512, 1024, 64, 128, 256]:
            for N in [512, 1024, 64, 128, 256]:
                #for K in [1024, 512, 64, 128, 256]:
                for K in [512, 1024, 64, 128, 256]:
                    # print(f'info: shape: {M} x {K} x {N}')
                    a = torch.randn((M, K), device="cuda", dtype=dtype)
                    b = torch.randn((K, N), device="cuda", dtype=dtype)

                    #a_1 = torch.randn((M-1, K-1), device="cuda", dtype=dtype)
                    #b_1 = torch.randn((K-1, N-1), device="cuda", dtype=dtype)


                    ms0 = run_normal_triton(a, b, M, K, N)
                    ms1 = run_torch(a, b, M, K, N)
                    #print(f'info: torch mm: {ms1}')
                    
                    #continue
                    triton_times2 = []
                    triton_times3 = []

                    for BLOCK_M in [256, 32, 64, 128]:
                        for BLOCK_K in [32, 64, 128]:
                            for BLOCK_N in [32, 64, 128]:
                                #print(f'info: BM: {BLOCK_M}, BK: {BLOCK_K}, BN: {BLOCK_N}')
                                if BLOCK_M > M or BLOCK_K > K or BLOCK_N > N:
                                    continue
                                
                                ms2 = torch.inf
                                try:
                                    ms2 = run_triton_block_mm(a, b, M, K, N, BLOCK_M, BLOCK_K, BLOCK_N)
                                except Exception as e:
                                    print(e)
                                    pass
                                ms3 = torch.inf
                                try:    
                                    pass
                                    ms3 = run_triton_naive_mm(a, b, M, K, N, BLOCK_M, BLOCK_K, BLOCK_N)
                                except:
                                    pass

                                #print(f'info: naive mm: {ms3}, block mm: {ms2}') 
                                
                                triton_times2.append((ms2, (BLOCK_M, BLOCK_K, BLOCK_N)))
                                triton_times3.append((ms3, (BLOCK_M, BLOCK_K, BLOCK_N)))
                                #sys.exit(1)

                    triton_times2.sort(key=lambda x: x[0])
                    triton_times3.sort(key=lambda x: x[0])
                    
                    print(f'{M} x {K} x {N}', end='; ')
                    print(f'{ms0:.4f}; {ms1:.4f}', end='; ')
                    for i in range(1):
                        ms, blocks = triton_times2[i]
                        print(f'{ms:.4f}; {blocks}', end='; ')

                        ms, blocks = triton_times3[i]
                        print(f'{ms:.4f}; {blocks}', end='; ')
                    print()
                    sys.stdout.flush()
                    #sys.exit(1)


def benchmark(shapes, dtypes):
    for dtype in dtypes:
        for shape in shapes:
            M, K, N = shape
            print(f'info: shape: {M} x {K} x {N}')
            a = torch.randn((M, K), device="cuda", dtype=dtype)
            b = torch.randn((K, N), device="cuda", dtype=dtype)

            #a_1 = torch.randn((M-1, K-1), device="cuda", dtype=dtype)
            #b_1 = torch.randn((K-1, N-1), device="cuda", dtype=dtype)

            ms0 = run_normal_triton(a, b, M, K, N)
            ms1 = run_torch(a, b, M, K, N)
            print(f'info: torch mm: {ms1}')
        
            triton_times2 = []
            triton_times3 = []

            for BLOCK_M in [32, 64, 128, 256]:
                for BLOCK_K in [32, 64, 128]:
                    for BLOCK_N in [32, 64, 128]:
                        #print(f'info: BM: {BLOCK_M}, BK: {BLOCK_K}, BN: {BLOCK_N}')
                        if BLOCK_M > M or BLOCK_K > K or BLOCK_N > N:
                            continue
                        
                        ms2 = torch.inf
                        try:
                            ms2 = run_triton_block_mm(a, b, M, K, N, BLOCK_M, BLOCK_K, BLOCK_N)
                        except Exception as e:
                            print(e)
                            pass
                        ms3 = torch.inf
                        try:    
                            pass
                            ms3 = run_triton_naive_mm(a, b, M, K, N, BLOCK_M, BLOCK_K, BLOCK_N)
                        except:
                            pass

                        print(f'info: naive mm: {ms3}, block mm: {ms2}') 
                        sys.stdout.flush()
                        
                        triton_times2.append((ms2, (BLOCK_M, BLOCK_K, BLOCK_N)))
                        triton_times3.append((ms3, (BLOCK_M, BLOCK_K, BLOCK_N)))
                        #sys.exit(1)

            triton_times2.sort(key=lambda x: x[0])
            triton_times3.sort(key=lambda x: x[0])
            
            print(f'{M} x {K} x {N}', end='; ')
            print(f'{ms0:.4f}; {ms1:.4f}', end='; ')
            for i in range(1):
                ms, blocks = triton_times2[i]
                print(f'{ms:.4f}; {blocks}', end='; ')

                ms, blocks = triton_times3[i]
                print(f'{ms:.4f}; {blocks}', end='; ')
            print()
            sys.stdout.flush()
            #sys.exit(1)


def run_real_shapes():
    benchmark(
        [(128, 9216, 4096), (128, 4096, 4096), (2048, 768, 768), (2048, 768, 3072)],
        [torch.float16, torch.float32]
    )
    

def run_regular_shapes():
    shapes = []
    for M in [512, 1024, 64, 128, 256]:
        for N in [512, 1024, 64, 128, 256]:
            for K in [512, 1024, 64, 128, 256]:
                shapes.append([M, K, N])
    benchmark(shapes, [torch.float16, torch.float32])

if __name__ == "__main__":
    check_block_format_utils()
    run_real_shapes()




