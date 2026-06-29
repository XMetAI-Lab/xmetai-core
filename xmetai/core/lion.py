import torch

__all__ = [
    "lion_update",
    "Lion",
]


def lion_update(grad, exp_avg, beta1=0.9, beta2=0.99):
    """Compute the Lion update direction and update the EMA buffer in place.

    Lion (EvoLved Sign Momentum) interpolates between the gradient and the
    momentum buffer to form the update, takes its sign, and then updates the
    momentum buffer with a separate decay. Using ``sign`` makes the update
    magnitude uniform across coordinates, which is the key behavioral
    difference from Adam-style optimizers.

    Args:
        grad: The current gradient tensor.
        exp_avg: The EMA momentum buffer, updated in place.
        beta1: Interpolation factor for forming the update direction.
        beta2: Decay factor for the momentum buffer.

    Returns:
        The sign-based update tensor (same shape as ``grad``).
    """
    update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1).sign_()
    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
    return update


class Lion(torch.optim.Optimizer):
    """Lion - EvoLved Sign Momentum.

    https://arxiv.org/abs/2302.06675

    Lion was discovered via symbolic program search and is both simpler and more
    memory-efficient than AdamW: it only tracks a single momentum buffer (no
    second-moment estimate), halving the optimizer state. The update is the sign
    of an interpolation between the gradient and the momentum, which yields a
    fixed update magnitude per coordinate and tends to act as a strong implicit
    regularizer.

    Because the update magnitude is decoupled from the gradient scale, Lion
    usually needs a learning rate roughly 3-10x smaller than AdamW and a
    correspondingly larger (decoupled) weight decay.

    Arguments:
        params: Iterable of parameters or parameter groups to optimize.
        lr: The learning rate.
        betas: Coefficients ``(beta1, beta2)`` used for the update direction and
            the momentum EMA respectively.
        weight_decay: Decoupled (AdamW-style) weight decay.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                # Decoupled weight decay.
                p.mul_(1 - group["lr"] * group["weight_decay"])
                update = lion_update(p.grad, state["exp_avg"], beta1=beta1, beta2=beta2)
                p.add_(update, alpha=-group["lr"])

        return loss
