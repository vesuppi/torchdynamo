import triton
from torchinductor.triton_ops.blocksparse.sum import sum
from torchinductor.triton_ops.blocksparse.exp import exp
from torchinductor.triton_ops.blocksparse.div import div

def softmax(x_mask, x_data):
    mask, t0 = exp(x_mask, x_data)
    t1 = sum(mask, t0, axis=1)
    mask, t2 = div(mask, t0, t1)
    return (mask, t2)