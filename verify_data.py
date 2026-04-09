"""Sanity-check data pipeline: build loaders and print batch shape + value range."""
import config
from dataset import build_dataloaders

def main():
    config.ensure_dirs()
    try:
        train_loader, val_loader, test_loader = build_dataloaders(
            batch_size=4,
            num_workers=0,
            img_size=config.IMG_SIZE,
        )
    except FileNotFoundError as e:
        print("Data not found:", e)
        print("Download the dataset and place it under data/ as in README.")
        return
    x, y = next(iter(train_loader))
    print("Train batch: x shape", x.shape, "dtype", x.dtype, "min", x.min().item(), "max", x.max().item())
    print("Train batch: y shape", y.shape, "unique", y.unique().tolist())
    xv, yv = next(iter(val_loader))
    print("Val batch:   x shape", xv.shape)
    print("Len train/val/test:", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    print("Data pipeline OK.")

if __name__ == "__main__":
    main()
