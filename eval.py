"""
Evaluate a trained model on the test set: Macro-F1, Precision, Recall, AUROC, confusion matrix.
Saves results to JSON and optional Markdown.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

import config
from config import ensure_dirs, CLASS_NAMES
from dataset import build_dataloaders
from models import build_model
from metrics import compute_metrics
from utils import get_device


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--model", type=str, default=None, help="Model name (default: from checkpoint)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    set_seed(config.SEED)
    ensure_dirs()
    out_dir = Path(args.output_dir or config.RESULTS_DIR)
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
        batch_size=args.batch_size or config.BATCH_SIZE,
        num_workers=0,
        img_size=config.IMG_SIZE,
    )

    all_preds = []
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Test", leave=False):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
            all_scores.append(probs)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_score = np.concatenate(all_scores, axis=0)

    metrics = compute_metrics(y_true, y_pred, y_score=y_score, num_classes=config.NUM_CLASSES)

    # Save JSON
    ckpt_stem = Path(args.checkpoint).stem
    json_path = out_dir / f"metrics_{ckpt_stem}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {json_path}")

    # Print and optional Markdown
    print("Test results:")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:       {metrics['macro_f1']:.4f}")
    print(f"  Macro-Precision:{metrics['macro_precision']:.4f}")
    print(f"  Macro-Recall:   {metrics['macro_recall']:.4f}")
    print(f"  AUROC:          {metrics['auroc']:.4f}")
    print("Confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))

    md_path = out_dir / f"metrics_{ckpt_stem}.md"
    with open(md_path, "w") as f:
        f.write("# Test Set Metrics\n\n")
        f.write(f"Checkpoint: `{args.checkpoint}`\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
        f.write(f"| Macro-F1 | {metrics['macro_f1']:.4f} |\n")
        f.write(f"| Macro-Precision | {metrics['macro_precision']:.4f} |\n")
        f.write(f"| Macro-Recall | {metrics['macro_recall']:.4f} |\n")
        f.write(f"| AUROC | {metrics['auroc']:.4f} |\n\n")
        f.write("## Confusion Matrix\n\n")
        f.write("|  | " + " | ".join(CLASS_NAMES) + " |\n")
        f.write("|--|" + "|".join(["---"] * len(CLASS_NAMES)) + "|\n")
        for i, row in enumerate(metrics["confusion_matrix"]):
            f.write(f"| **{CLASS_NAMES[i]}** | " + " | ".join(map(str, row)) + " |\n")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
