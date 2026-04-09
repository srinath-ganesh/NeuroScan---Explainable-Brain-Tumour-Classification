"""
Grad-CAM: gradient-weighted class activation mapping.
Uses last convolutional layer; alpha_k = GAP(dY^c/dA^k); CAM = ReLU(sum_k alpha_k * A^k).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def _get_last_conv_module(model: nn.Module, name: str):
    """Return the last conv layer with spatial extent for Grad-CAM."""
    name = name.lower()
    if "efficientnet" in name:
        # Use last Conv2d in features (not classifier's 1x1)
        last_conv = None
        for m in model.features.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv
    if "resnet" in name:
        return model.layer4
    return None


class GradCAM:
    def __init__(self, model: nn.Module, model_name: str = "efficientnet_b0"):
        self.model = model
        self.model_name = model_name
        self._target_layer = _get_last_conv_module(model, model_name)
        self._activations = None
        self._gradients = None
        self._hook_handles = []

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def _register_hooks(self):
        if self._target_layer is None:
            raise RuntimeError("Could not find last conv layer for this model")
        self._hook_handles.append(
            self._target_layer.register_forward_hook(self._save_activation)
        )
        self._hook_handles.append(
            self._target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        x: (1, C, H, W). Returns heatmap (H, W) in [0, 1] range, same spatial size as input x.
        """
        self.model.eval()
        self._register_hooks()
        self._activations = None
        self._gradients = None

        x = x.requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()

        A = self._activations  # (1, K, h, w)
        g = self._gradients    # (1, K, h, w)
        self._remove_hooks()

        weights = g.mean(dim=(2, 3))  # (1, K)
        cam = (weights[:, :, None, None] * A).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam
