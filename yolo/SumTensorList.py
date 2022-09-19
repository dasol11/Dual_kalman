import torch

def sum_tensor(li):
    a = torch.zeros((0, 6), device='cuda')
    for i in li:
        a = torch.cat([a, i], dim=0)

    return a