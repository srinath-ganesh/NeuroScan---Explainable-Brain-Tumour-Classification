"""
Focal Loss for multi-class classification with optional class weights.
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Label-smoothed focal loss with an auxiliary cross-entropy term.

    The blended objective keeps gradients alive longer than pure focal loss,
    which helps both classification and Grad-CAM stability.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha=None,
        reduction: str = "mean",
        label_smoothing: float = 0.05,
        focal_weight: float = 0.7,
        ce_weight: float = 0.3,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor of shape (num_classes,) or None
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        self.temperature = temperature

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), targets: (B,) long
        if logits.ndim != 2:
            raise ValueError(f"Expected logits with shape (B, C), got {tuple(logits.shape)}")

        num_classes = logits.shape[1]
        temp = max(float(self.temperature), 1e-6)
        scaled_logits = logits / temp

        log_probs = F.log_softmax(scaled_logits, dim=1)
        probs = log_probs.exp()

        with torch.no_grad():
            target_dist = F.one_hot(targets, num_classes=num_classes).float()
            if self.label_smoothing > 0:
                smooth = self.label_smoothing / num_classes
                target_dist = target_dist * (1.0 - self.label_smoothing) + smooth

        ce = -(target_dist * log_probs).sum(dim=1)

        pt = (probs * target_dist).sum(dim=1).clamp(min=1e-8, max=1.0)
        focal = ((1.0 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            class_weights = torch.as_tensor(self.alpha, device=logits.device, dtype=logits.dtype)
            if class_weights.numel() != num_classes:
                raise ValueError(f"alpha must have {num_classes} values, got {class_weights.numel()}")
            sample_weights = class_weights[targets]
            ce = ce * sample_weights
            focal = focal * sample_weights

        loss = self.focal_weight * focal + self.ce_weight * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
