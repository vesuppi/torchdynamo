import torch

def batchnorm(x):
    t0 = torch.sum(x, axis=0)
    