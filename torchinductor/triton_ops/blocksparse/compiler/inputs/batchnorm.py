import torch

def layernorm(x, N):
    mu = torch.sum(x, axis=-1) / N
    t0 = x - mu
    t1 = t0 * t0
    t2 = torch.sum(t1, axis=-1) / N
    sigma = torch.sqrt(t2+1e-5)
    y = (x - mu) / sigma
    return y
    