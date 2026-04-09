"""
Single-image inference + Grad-CAM explainability pipeline.

Usage:
    python inference.py path/to/random_mri.jpg
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T

import config
from config import ensure_dirs, CLASS_NAMES
from models import build_model
from explainability import GradCAM
from utils import get_device


def get_image_transform():
    """
    Standard ImageNet-style transforms:
    - Resize to 224x224
    - ToTensor
    - Normalize with ImageNet mean/std
    """
    return T.Compose(
        [
            T.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Build EfficientNet-B0 with 4-class head
    model = build_model("efficientnet_b0", pretrained=False, num_classes=config.NUM_CLASSES)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    Returns RGB overlay in [0,1].
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    heatmap_uint8 = np.uint8(255 * heatmap)
    cmap = plt.cm.jet(heatmap_uint8)[..., :3]  # (H,W,3)
    out = (1 - alpha) * img + alpha * cmap
    return np.clip(out, 0, 1)


def main():
    parser = argparse.ArgumentParser(description="Single-image inference with Grad-CAM explainability.")
    parser.add_argument("image_path", type=str, help="Path to MRI image (jpg/png).")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    ensure_dirs()
    explain_dir = config.RESULTS_DIR / "explainability"
    explain_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # Load model with SAM checkpoint
    ckpt_path = config.CHECKPOINTS_DIR / "best_efficientnet_b0_sam.pt"
    model = load_model(ckpt_path, device)

    # Load and preprocess image
    pil_img = Image.open(img_path).convert("RGB")
    transform = get_image_transform()
    x = transform(pil_img)  # (3,H,W)
    x = x.unsqueeze(0).to(device)  # (1,3,H,W)

    # Forward pass and softmax probabilities
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx = int(probs.argmax())
    top_class = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx]) * 100.0

    # Print clean, formatted prediction
    print("\n=== Inference Result ===")
    print(f"Image: {img_path}")
    print(f"Predicted tumor type: {top_class} ({top_conf:.2f}% confidence)")
    print("\nClass probabilities:")
    for i, p in sorted(enumerate(probs), key=lambda t: t[1], reverse=True):
        print(f"  {CLASS_NAMES[i]:<10}: {p * 100.0:6.2f}%")

    # Grad-CAM for this image
    gradcam = GradCAM(model, model_name="efficientnet_b0")
    cam = gradcam(x, class_idx=top_idx)  # (H,W) in [0,1]

    # Prepare original image for overlay (convert back to [0,1] grayscale)
    # Use the preprocessed tensor but undo channel dimension & normalization effect for display
    # Here we just take mean over channels after inverse-normalizing approximately.
    inv_mean = np.array([0.485, 0.456, 0.406])
    inv_std = np.array([0.229, 0.224, 0.225])
    x_np = x[0].detach().cpu().numpy()  # (3,H,W)
    x_np = (x_np * inv_std[:, None, None]) + inv_mean[:, None, None]
    x_np = np.clip(x_np, 0.0, 1.0)
    base_img = x_np.mean(axis=0)  # (H,W)

    overlay = overlay_heatmap(base_img, cam, alpha=0.5)

    # Save combined image
    out_path = explain_dir / "latest_inference.jpg"
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"{top_class} ({top_conf:.2f}% confidence)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved Grad-CAM overlay to: {out_path}")

    # Show pop-up window
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"{top_class} ({top_conf:.2f}% confidence)")
    plt.tight_layout()
    plt.show()  # blocks until window closed


if __name__ == "__main__":
    main()

