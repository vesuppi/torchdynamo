import torch

def softmax(x):
    max = torch.max(x, axis=-1)
    normalized = x - max[:, None]
    e = torch.exp(normalized)
    s = torch.sum(e, axis=-1)
    y = torch.div(e, s[:, None])
    return y
