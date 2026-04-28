"""
Training script: baseline (Adam) or SAM. Saves best model by validation Macro-F1.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from tqdm.auto import tqdm

import config
from config import ensure_dirs
from dataset import build_dataloaders
from models import build_model
from losses import FocalLoss
from metrics import compute_metrics
from monitoring import append_jsonl, build_reference_samples, save_gradcam_epoch_snapshot
from sam import sam_step
from utils import get_device


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, loss_fn, optimizer, device, use_sam, rho, max_batches=None):
    model.train()
    total_loss = 0.0
    n = 0
    for bi, (x, y) in enumerate(tqdm(train_loader, desc="Train", leave=False)):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        def closure():
            logits = model(x)
            return loss_fn(logits, y)

        if use_sam:
            loss = sam_step(model, optimizer, closure, rho=rho)
        else:
            optimizer.zero_grad()
            loss = closure()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n if n else 0.0


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    for x, y in tqdm(val_loader, desc="Val", leave=False):
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu()
        scores = torch.softmax(logits, dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(y)
        all_scores.append(scores)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    scores = torch.cat(all_scores)
    return compute_metrics(labels.numpy(), preds.numpy(), scores.numpy(), num_classes=config.NUM_CLASSES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--epochs", type=int, default=None, help="Defaults to config.NUM_EPOCHS")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--use_sam", action="store_true")
    parser.add_argument("--rho", type=float, default=None, help="SAM neighborhood radius")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--monitor_dir", type=str, default=None, help="Directory for history logs and Grad-CAM snapshots")
    parser.add_argument("--gradcam_log_every", type=int, default=1, help="Log Grad-CAM snapshots every N epochs")
    parser.add_argument("--gradcam_samples_per_class", type=int, default=1, help="Number of fixed samples per class for epoch tracking")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches per epoch (for quick test)")
    args = parser.parse_args()

    set_seed(config.SEED)
    ensure_dirs()
    ckpt_dir = Path(args.checkpoint_dir or config.CHECKPOINTS_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs or config.NUM_EPOCHS
    batch_size = args.batch_size or config.BATCH_SIZE
    lr = args.lr or config.LEARNING_RATE
    rho = args.rho if args.rho is not None else config.SAM_RHO
    monitor_dir = Path(args.monitor_dir or config.MONITORING_DIR)
    gradcam_dir = monitor_dir / "gradcam_epochs"
    history_path = monitor_dir / "training_history.jsonl"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_dataloaders(
        batch_size=batch_size,
        num_workers=args.num_workers,
        img_size=config.IMG_SIZE,
    )
    device = get_device()
    print(f"Using device: {device}")
    model = build_model(args.model, pretrained=True, num_classes=config.NUM_CLASSES).to(device)
    loss_fn = FocalLoss(gamma=config.FOCAL_LOSS_GAMMA, alpha=config.FOCAL_LOSS_ALPHA)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    gradcam_probe_samples = build_reference_samples(
        val_loader,
        device=device,
        per_class=max(1, args.gradcam_samples_per_class),
    )

    suffix = "sam" if args.use_sam else "baseline"
    best_f1 = -1.0
    for ep in range(epochs):
        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device,
            use_sam=args.use_sam, rho=rho, max_batches=args.max_batches,
        )
        metrics = validate(model, val_loader, device)
        val_f1 = metrics["macro_f1"]
        print(
            f"Epoch {ep+1}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_macro_f1={val_f1:.4f}  val_acc={metrics['accuracy']:.4f}  "
            f"val_auroc={metrics.get('auroc', float('nan')):.4f}"
        )

        gradcam_summary = {}
        if (ep + 1) % max(1, args.gradcam_log_every) == 0:
            gradcam_summary = save_gradcam_epoch_snapshot(
                model=model,
                samples=gradcam_probe_samples,
                epoch=ep + 1,
                model_name=args.model,
                output_dir=gradcam_dir,
            )

        history_record = {
            "epoch": ep + 1,
            "epochs": epochs,
            "model": args.model,
            "use_sam": bool(args.use_sam),
            "train_loss": float(train_loss),
            **metrics,
            **gradcam_summary,
            "best_so_far": bool(val_f1 > best_f1),
        }
        append_jsonl(history_path, history_record)

        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = ckpt_dir / f"best_{args.model}_{suffix}.pt"
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "config": {
                    "model": args.model,
                    "use_sam": args.use_sam,
                    "rho": rho,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "monitor_dir": str(monitor_dir),
                },
            }, ckpt_path)
            print(f"  -> saved {ckpt_path}")

    print(f"Done. Best val Macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
