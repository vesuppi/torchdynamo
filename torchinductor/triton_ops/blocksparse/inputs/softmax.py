import torch

def softmax(x):
    max = torch.max(x, 0)
    normalized = x - max
    e = torch.exp(normalized)
    s = torch.sum(e, axis=0)
    y = torch.div(e, s)
    return y
