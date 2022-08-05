import sys
import ast
import astunparse
from ast import *
from typing import Any

## template parameters
func = ''
statements = '\n'
compressed = '0'
load_x_nnz = 'len = tl.load(x_cols + 2*outer_m+1)'

class FuncLister(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
        global func, compressed
        func = node.name
        if func == 'softmax':
            compressed = '-float("inf")'
        self.generic_visit(node)

    def visit_Assign(self, node: Assign) -> Any:
        global statements
        stmt = astunparse.unparse(node)
        tl_stmt = stmt.strip().replace('torch', 'tl')\
                              .replace('div', 'fdiv')\
                              .replace('axis=-1', 'axis=0')\
                              .replace('[(:, None)]', '')
        statements += ' '*4 + tl_stmt + '\n'


if __name__ == '__main__':
    inp = sys.argv[1]
    content = open(inp).read()
    tree = ast.parse(content)
    FuncLister().visit(tree)



unary_kernel = \
f'''
import math
import torch
import triton
import triton.language as tl
from torchinductor.triton_ops.blocksparse.utils import FastBCSR

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

@triton.jit
def _{func}_kernel(x_rowptrs, x_cols, x_data, y_data, 
                stride_xz,
                M: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr,
                TM: tl.constexpr, TN: tl.constexpr, 
                ):
    inner_m = tl.program_id(0)
    outer_m = tl.program_id(1)
    z = tl.program_id(2)
    block_size = BM * BN

    data_offset_for_this_row = tl.load(x_rowptrs+outer_m)
    offsets = stride_xz * z
    offsets += data_offset_for_this_row * block_size
    
    col_start = tl.load(x_cols + 2*outer_m)
    {load_x_nnz}

    block_n = tl.arange(0, N) // BN
    lane_n = tl.arange(0, N) % BN

    offsets += block_n * block_size + lane_n
    offsets += inner_m * BN
    x_ptrs = x_data + offsets 

    mask = block_n < len
    x = tl.load(x_ptrs, mask=mask, other={compressed})
    {statements}
    y_ptrs = y_data + offsets
    tl.store(y_ptrs, y, mask=mask)

def {func}(x: FastBCSR):
    rowptrs, cols, fastcols, x_vals = x.rowptrs, x.cols, x.fastcols, x.vals
    Z, M, N, BM, BN = x.Z, x.M, x.N, x.BM, x.BN
    m = M // BM
    y_data = torch.empty_like(x_vals)
    grid = (BM, m, Z)
    _{func}_kernel[grid](
            rowptrs, fastcols, x_vals, y_data,
            x_vals.stride(0),
            M, N, BM, BN, 1, BN,
            num_warps=num_warps(N)
        )
    
    f = FastBCSR(Z, M, N, BM, BN, rowptrs, cols, fastcols, y_data, x.is_blocks_dense, default=0)
    return f
'''

print(unary_kernel)