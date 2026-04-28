"""Training monitoring helpers.

This module centralizes:
- history logging to JSONL
- Grad-CAM snapshot generation across epochs
- lightweight image overlays used by the dashboard and training loop
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CLASS_NAMES
from explainability import GradCAM


def tensor_to_display_image(x: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor (C,H,W) or (1,C,H,W) into a displayable [0,1] grayscale RGB image."""
    if x.ndim == 4:
        x = x[0]
    x_np = x.detach().cpu().numpy()
    if x_np.ndim != 3:
        raise ValueError(f"Expected tensor with 3 dims after batching, got {x_np.shape}")
    x_np = x_np.mean(axis=0)
    x_np = np.nan_to_num(x_np)
    if x_np.max() > x_np.min():
        x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min())
    x_np = np.clip(x_np, 0.0, 1.0)
    return x_np


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a jet heatmap onto a grayscale image."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    cmap = plt.cm.jet(heatmap)[..., :3]
    out = (1 - alpha) * img + alpha * cmap
    return np.clip(out, 0, 1)


def append_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_reference_samples(
    loader: Iterable,
    device: torch.device,
    per_class: int = 1,
) -> List[Tuple[torch.Tensor, int]]:
    """Collect a small, fixed set of validation/test samples for epoch-by-epoch Grad-CAM tracking."""
    quota = {idx: 0 for idx in range(len(CLASS_NAMES))}
    samples: List[Tuple[torch.Tensor, int]] = []
    for x, y in loader:
        for i in range(x.size(0)):
            label = int(y[i].item())
            if quota[label] >= per_class:
                continue
            samples.append((x[i : i + 1].to(device), label))
            quota[label] += 1
            if all(v >= per_class for v in quota.values()):
                return samples
    return samples


def save_gradcam_epoch_snapshot(
    model: torch.nn.Module,
    samples: Sequence[Tuple[torch.Tensor, int]],
    epoch: int,
    model_name: str,
    output_dir: Path,
) -> Dict:
    """Generate a small Grad-CAM report + image grid for a fixed sample set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gradcam = GradCAM(model, model_name=model_name)

    if not samples:
        return {"gradcam_samples": 0, "gradcam_mean": float("nan"), "gradcam_max": float("nan"), "gradcam_coverage": float("nan")}

    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.2 * n))
    if n == 1:
        axes = np.array([axes])

    stats = []
    for row, (x, y_true) in enumerate(samples):
        x = x.to(next(model.parameters()).device)
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            y_pred = int(probs.argmax().item())
            pred_conf = float(probs[y_pred].item())

        cam = gradcam(x, class_idx=y_pred)
        base_img = tensor_to_display_image(x)
        overlay = overlay_heatmap(base_img, cam, alpha=0.5)

        stats.append(
            {
                "true": int(y_true),
                "pred": int(y_pred),
                "confidence": pred_conf,
                "cam_mean": float(np.mean(cam)),
                "cam_max": float(np.max(cam)),
                "cam_coverage": float(np.mean(cam > 0.5)),
            }
        )

        axes[row, 0].imshow(base_img, cmap="gray")
        axes[row, 0].set_title(f"Input | true={CLASS_NAMES[y_true]} pred={CLASS_NAMES[y_pred]}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(
            f"Grad-CAM | conf={pred_conf:.2f} mean={stats[-1]['cam_mean']:.3f} coverage={stats[-1]['cam_coverage']:.3f}"
        )
        axes[row, 1].axis("off")

    fig.suptitle(f"Epoch {epoch:03d} Grad-CAM snapshot", y=0.995)
    fig.tight_layout()
    out_path = output_dir / f"epoch_{epoch:03d}_gradcam.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "gradcam_samples": n,
        "gradcam_mean": float(np.mean([s["cam_mean"] for s in stats])),
        "gradcam_max": float(np.mean([s["cam_max"] for s in stats])),
        "gradcam_coverage": float(np.mean([s["cam_coverage"] for s in stats])),
        "gradcam_preview": str(out_path),
        "gradcam_sample_details": stats,
    }
    return summary
