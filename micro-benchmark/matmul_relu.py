import sre_compile
import torch
from benchmark_helper import time_with_torch_timer
from torchinductor.triton_ops.matmul import matmul_out as triton_mm_out
from torchinductor.triton_ops.matmul import matmul as triton_mm
import torchdynamo
import torchinductor.config as config

config.triton.use_mm = True

@torchdynamo.optimize("inductor", nopython=True)
def inductor_mm(a, b):
    return torch.mm(a, b)

def torch_mm_relu(a, b):
    return torch.nn.functional.relu(torch.matmul(a, b))

def torch_mm(a, b):
    return torch.matmul(a, b)


if __name__ == "__main__":
    # Real shapes from torchbench
    a_shapes = [ [2048, 768], [64, 1280], [2048, 768], [32, 2048], [1, 39200], [128, 3072], [16, 1280]]
    b_shapes = [ [768, 3072], [1280, 1000], [768, 768], [2048, 1000], [39200, 50], [3072, 1000], [1280, 1000]]

    # Artificial larger shapes
    a_shapes += [[10240, 512], [10240, 1024]]
    b_shapes += [[512, 10240], [1024, 10240]]

    for i in range(len(a_shapes)):
        a_shape = a_shapes[i]
        b_shape = b_shapes[i]
        print("Shape:", a_shape, 'x', b_shape)
        a = torch.randn(a_shape, device='cuda', dtype=torch.float16)
        b = torch.randn(b_shape, device='cuda', dtype=a.dtype)

        time_with_torch_timer(torch_mm, (a, b), string_id="torch mm")
        time_with_torch_timer(torch_mm_relu, (a, b), string_id="torch mm + relu")
        time_with_torch_timer(inductor_mm, (a, b), string_id="inductor mm")
        




# Results preview
'''
Shape: [2048, 768] x [768, 3072]
torch mm         mean: 0.0593 ms
torch mm + relu  mean: 0.0764 ms
inductor mm      mean: 0.0680 ms
Shape: [64, 1280] x [1280, 1000]
torch mm         mean: 0.0211 ms
torch mm + relu  mean: 0.0306 ms
inductor mm      mean: 0.0268 ms
Shape: [2048, 768] x [768, 768]
torch mm         mean: 0.0195 ms
torch mm + relu  mean: 0.0293 ms
inductor mm      mean: 0.0276 ms
Shape: [32, 2048] x [2048, 1000]
torch mm         mean: 0.0198 ms
torch mm + relu  mean: 0.0304 ms
inductor mm      mean: 0.0274 ms
Shape: [1, 39200] x [39200, 50]
torch mm         mean: 0.0146 ms
torch mm + relu  mean: 0.0245 ms
inductor mm      mean: 0.0305 ms
Shape: [128, 3072] x [3072, 1000]
torch mm         mean: 0.0206 ms
torch mm + relu  mean: 0.0321 ms
inductor mm      mean: 0.0322 ms
Shape: [16, 1280] x [1280, 1000]
torch mm         mean: 0.0239 ms
torch mm + relu  mean: 0.0309 ms
inductor mm      mean: 0.0272 ms
Shape: [10240, 512] x [512, 10240]
torch mm         mean: 0.4678 ms
torch mm + relu  mean: 0.7760 ms
inductor mm      mean: 0.5435 ms
Shape: [10240, 1024] x [1024, 10240]
torch mm         mean: 0.9312 ms
torch mm + relu  mean: 1.2112 ms
inductor mm      mean: 0.9921 ms
'''
