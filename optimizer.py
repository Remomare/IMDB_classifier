import torch

def Adam(parameters, lr):
    return torch.optim.Adam(parameters, lr)