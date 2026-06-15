import torch


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )