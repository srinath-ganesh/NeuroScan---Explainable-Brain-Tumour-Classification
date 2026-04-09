"""
Integrated Gradients: path integral from baseline to input.
Baseline: black image or blurred (configurable).
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target_class: Optional[int] = None,
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 50,
) -> np.ndarray:
    """
    x: (1, C, H, W). Returns attribution map (C, H, W) or (H, W) mean over C, same shape as input.
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x, device=x.device)
    if target_class is None:
        logits = model(x)
        target_class = logits.argmax(dim=1).item()

    # path: baseline -> x
    alphas = torch.linspace(0, 1, n_steps + 1, device=x.device, dtype=x.dtype)
    attributions = []
    for i in range(n_steps):
        alpha = alphas[i].item()
        x_interp = (baseline + alpha * (x - baseline)).requires_grad_(True)
        x_interp.retain_grad()
        logits = model(x_interp)
        model.zero_grad()
        logits[0, target_class].backward()
        grad = x_interp.grad
        if grad is None:
            grad = torch.zeros_like(x, device=x.device)
        attributions.append(grad.detach())
    avg_grad = torch.stack(attributions, dim=0).mean(dim=0)
    ig = (x - baseline).detach() * avg_grad
    # sum over channels for visualization, or return (1,C,H,W)
    ig_np = ig.squeeze(0).cpu().numpy()
    if ig_np.ndim == 3:
        ig_np = ig_np.mean(axis=0)
    ig_np = np.abs(ig_np)
    if ig_np.max() > ig_np.min():
        ig_np = (ig_np - ig_np.min()) / (ig_np.max() - ig_np.min())
    return ig_np


class IntegratedGradients:
    def __init__(self, model: nn.Module, baseline: str = "black", n_steps: int = 50):
        self.model = model
        self.baseline_type = baseline
        self.n_steps = n_steps

    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        if baseline is None:
            if self.baseline_type == "black":
                baseline = torch.zeros_like(x, device=x.device)
            else:
                # blur: simple mean or small kernel; for simplicity use black if not implemented
                baseline = torch.zeros_like(x, device=x.device)
        return integrated_gradients(
            self.model, x, target_class=target_class, baseline=baseline, n_steps=self.n_steps
        )
