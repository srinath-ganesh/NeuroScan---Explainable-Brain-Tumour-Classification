"""
End-to-end prediction + Grad-CAM pipeline for a single MRI image.
Uses PIL and torchvision.transforms (Resize 256, CenterCrop 224) for safe handling of internet images.

Usage:
    python pipeline.py path/to/image.jpg
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T
import matplotlib.pyplot as plt

import config
from config import ensure_dirs, CLASS_NAMES
from models import build_model
from explainability import GradCAM
from utils import get_device


def get_imagenet_transform() -> T.Compose:
    """
    Standard ImageNet-style transforms; CenterCrop(224) safely removes
    weird internet borders without stretching the brain.
    """
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_sam_model(device: torch.device) -> torch.nn.Module:
    ckpt_path = config.CHECKPOINTS_DIR / "best_efficientnet_b0_sam.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {ckpt_path}")
    model = build_model("efficientnet_b0", pretrained=False, num_classes=config.NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()
    return model


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    img: (H,W) or (H,W,3) in [0,1]
    heatmap: (H,W) in [0,1]
    Returns overlay RGB image in [0,1].
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    heatmap_uint8 = np.uint8(255 * heatmap)
    cmap = plt.cm.jet(heatmap_uint8)[..., :3]
    out = (1 - alpha) * img + alpha * cmap
    return np.clip(out, 0, 1)


def main():
    parser = argparse.ArgumentParser(description="End-to-end prediction + Grad-CAM for a single MRI image.")
    parser.add_argument("image_path", type=str, help="Path to MRI image (jpg/png).")
    args = parser.parse_args()

    raw_path = Path(args.image_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Image not found: {raw_path}")

    ensure_dirs()
    explain_dir = config.RESULTS_DIR / "explainability"
    explain_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model = load_sam_model(device)

    # Load image with PIL and apply transforms
    pil_img = Image.open(raw_path).convert("RGB")
    transform = get_imagenet_transform()
    x = transform(pil_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx = int(probs.argmax())
    top_class = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx]) * 100.0

    # Diagnosis report
    print("\n=== Diagnosis Report ===")
    print(f"Image:          {raw_path}")
    print(f"Predicted type: {top_class} ({top_conf:.2f}% confidence)")
    print("\nClass probabilities:")
    for i, p in sorted(enumerate(probs), key=lambda t: t[1], reverse=True):
        print(f"  {CLASS_NAMES[i]:<10}: {p * 100.0:6.2f}%")

    # Grad-CAM
    gradcam = GradCAM(model, model_name="efficientnet_b0")
    cam = gradcam(x, class_idx=top_idx)

    # Inverse-normalize for visualization
    inv_mean = np.array([0.485, 0.456, 0.406])
    inv_std = np.array([0.229, 0.224, 0.225])
    x_np = x[0].detach().cpu().numpy()  # (3,H,W)
    x_np = (x_np * inv_std[:, None, None]) + inv_mean[:, None, None]
    x_np = np.clip(x_np, 0.0, 1.0)
    base_img = x_np.mean(axis=0)  # (H,W)

    overlay = overlay_heatmap(base_img, cam, alpha=0.5)

    # Save final overlay
    out_path = explain_dir / "latest_pipeline_result.jpg"
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"{top_class} ({top_conf:.2f}% confidence)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved Grad-CAM overlay to: {out_path}")

    # Pop-up window (hold until user closes)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"{top_class} ({top_conf:.2f}% confidence)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
