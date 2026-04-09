"""
Generate Grad-CAM and Integrated Gradients visualizations for selected test samples.
Saves overlay images to output_dir.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

import config
from config import ensure_dirs, CLASS_NAMES
from dataset import build_dataloaders
from models import build_model
from explainability import GradCAM, IntegratedGradients
from utils import get_device


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """img (H,W) or (H,W,3), heatmap (H,W). Returns RGB overlay."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = plt.cm.jet(heatmap)[:, :, :3]
    overlay = (1 - alpha) * img + alpha * heatmap
    return np.clip(overlay, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=8, help="Total samples to visualize (2 per class)")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(config.SEED)
    ensure_dirs()
    out_dir = Path(args.output_dir or config.RESULTS_DIR) / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "config" in ckpt:
        model_name = ckpt["config"].get("model", "efficientnet_b0")
    else:
        model_name = args.model or "efficientnet_b0"

    device = get_device()
    print(f"Using device: {device}")
    model = build_model(model_name, pretrained=False, num_classes=config.NUM_CLASSES)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()

    _, _, test_loader = build_dataloaders(
        batch_size=1,
        num_workers=args.num_workers,
        img_size=config.IMG_SIZE,
    )
    # Collect one sample per class then fill up to num_samples
    samples_per_class = max(1, args.num_samples // config.NUM_CLASSES)
    collected = {c: [] for c in range(config.NUM_CLASSES)}
    for x, y in tqdm(test_loader, desc="Collect samples", leave=False):
        if all(len(collected[c]) >= samples_per_class for c in range(config.NUM_CLASSES)):
            break
        lab = y.item()
        if len(collected[lab]) < samples_per_class:
            collected[lab].append((x.clone(), y.item()))

    flat = []
    for c in range(config.NUM_CLASSES):
        for x, y in collected[c]:
            flat.append((x, y))
    if not flat:
        print("No test samples found. Ensure dataset is loaded.")
        return

    gradcam = GradCAM(model, model_name=model_name)
    ig = IntegratedGradients(model, baseline=config.IG_BASELINE, n_steps=config.IG_N_STEPS)

    for idx, (x, y_true) in enumerate(tqdm(flat, desc="Explain", leave=False)):
        x = x.to(device)
        with torch.no_grad():
            logits = model(x)
            y_pred = logits.argmax(dim=1).item()

        cam = gradcam(x, class_idx=y_pred)
        ig_map = ig(x, target_class=y_pred)

        img = x[0].detach().cpu().numpy().mean(axis=0)  # (H, W)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")

        axes[2].imshow(ig_map, cmap="hot")
        axes[2].set_title("Integrated Gradients")
        axes[2].axis("off")

        overlay = overlay_heatmap(img, cam, alpha=0.5)
        axes[3].imshow(overlay)
        axes[3].set_title(f"Overlay (True: {CLASS_NAMES[y_true]}, Pred: {CLASS_NAMES[y_pred]})")
        axes[3].axis("off")

        plt.tight_layout()
        out_path = out_dir / f"explain_{idx:02d}_true_{CLASS_NAMES[y_true]}_pred_{CLASS_NAMES[y_pred]}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")

    print(f"Done. Figures in {out_dir}")


if __name__ == "__main__":
    main()
