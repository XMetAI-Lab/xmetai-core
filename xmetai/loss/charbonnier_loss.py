import torch
__all__ = [ "charbonnier_loss"]

def charbonnier_loss(output, target, weight=None, eps=1e-6, reduction="mean"):
    if weight is None:
        loss = torch.sqrt(((output - target))**2 + eps)
    else:
        loss = torch.sqrt((weight * (output - target))**2 + eps)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

