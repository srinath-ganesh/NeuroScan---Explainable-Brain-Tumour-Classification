"""
Sharpness-Aware Minimization (SAM): two-step training step.
1. Ascent: compute epsilon that maximizes loss in L2-ball of radius rho.
2. Descent: forward at theta+epsilon, backward, restore theta, then optimizer.step() with gradients at theta+epsilon.
"""
import torch
from torch import Tensor
from typing import Callable, List


def sam_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[], Tensor],
    rho: float = 0.05,
) -> Tensor:
    """
    Perform one SAM update: ascent to get perturbed params, then descent at perturbed point.
    Returns the loss value (before ascent).
    """
    # First forward-backward to get gradients at theta
    optimizer.zero_grad()
    loss = loss_fn()
    loss.backward()

    grads = []
    params_with_grad = []
    for p in model.parameters():
        if p.grad is None:
            continue
        grads.append(p.grad.detach().flatten())
        params_with_grad.append(p)
    if not grads:
        optimizer.step()
        return loss.detach()
    grad_norm = torch.cat(grads).norm()
    if grad_norm < 1e-12:
        optimizer.step()
        return loss.detach()
    scale = rho / (grad_norm + 1e-12)

    # Save current theta
    saved = [p.detach().clone() for p in params_with_grad]

    # Ascent: theta <- theta + epsilon (epsilon = scale * grad)
    with torch.no_grad():
        for p in params_with_grad:
            p.add_(p.grad, alpha=scale)

    # Second forward-backward at theta + epsilon
    optimizer.zero_grad()
    loss_pert = loss_fn()
    loss_pert.backward()

    # Restore theta (gradients at theta+epsilon remain in .grad)
    with torch.no_grad():
        for p, s in zip(params_with_grad, saved):
            p.copy_(s)

    optimizer.step()
    return loss.detach()
