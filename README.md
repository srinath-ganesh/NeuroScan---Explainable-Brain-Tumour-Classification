# Brain Tumor Classification (EfficientNet + SAM) — DLCV Final Project

Robust and interpretable multi-class brain tumor classification from **2D MRI slices** using:
- **EfficientNet-B0** (primary) and **ResNet-50** (baseline comparator)
- **Focal Loss**
- **Sharpness-Aware Minimization (SAM)** optimizer
- **Grad-CAM** (plus Integrated Gradients utilities) for explainability

This repo is set up to run both:
- **Locally** (macOS Apple Silicon is supported via **MPS**), and
- **Google Colab** (CUDA GPU).

---

## What’s in / not in GitHub

This repo intentionally does **not** include:
- the Kaggle dataset (`data/`)
- training outputs (`outputs/`)
- large checkpoints (`*.pt`)
- local virtual environments (`venv/`)
- other unrelated coursework folders you mentioned (ignored by `.gitignore`)

Share checkpoints via Google Drive (as shown in Colab notebook) or another storage, and keep the repo lightweight.

---

## Dataset

- **Source**: Kaggle Brain Tumor MRI Dataset (4 classes): `glioma`, `meningioma`, `notumor`, `pituitary`
- **Expected layout**:

```text
data/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

### Download (optional; recommended)

If you have Kaggle CLI configured:

```bash
mkdir -p data
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data
cd data && unzip brain-tumor-mri-dataset.zip && cd ..
```

If the filename differs after download, list the directory and unzip the downloaded zip.

### Dataset link (if the download command ever changes)

- Main dataset page: `https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset`

---

## Optional: Kaggle_Mega_Test (OOD “internet” evaluation packs)

This repo may be evaluated on additional “mega test” datasets. **These are NOT committed to GitHub** (see `.gitignore`), but we document them so teammates can reproduce OOD testing.

Local folder (ignored by git):

```text
Kaggle_Mega_Test/
  Dataset_1_Sartaj/        (~3264 images)
  Dataset_2_Mohammad/      (~21672 images)
  Dataset_3_Shreya/        (~1311 images)
  Dataset_4_AbouAli/       (~10936 images)
  Downloaded Zips/         (original .zip downloads)
```

Known/likely sources:
- **Dataset_1_Sartaj**: Kaggle “Brain Tumor Classification (MRI)”  
  `https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri`
- CLI:
  ```bash
  kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri
  ```
- **Dataset_2_Mohammad**: Kaggle “Brain Tumor MRI Dataset” (Normal/ + Tumor/*tumor)  
  `https://www.kaggle.com/datasets/deeppythonist/brain-tumor-mri-dataset`
- CLI:
  ```bash
  kaggle datasets download -d deeppythonist/brain-tumor-mri-dataset
  ```
- **Dataset_3_Shreya**: Kaggle “Brain MRI Scans for Brain Tumor Classification”  
  `https://www.kaggle.com/datasets/shreyagupta19/brain-mri-scans-for-brain-tumor-classification`
- CLI:
  ```bash
  kaggle datasets download -d shreyagupta19/brain-mri-scans-for-brain-tumor-classification
  ```
- **Dataset_4_AbouAli**: Kaggle “MRI Brain Tumor Dataset 4-Class (7023 images)”  
  `https://www.kaggle.com/datasets/mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images`
- CLI:
  ```bash
  kaggle datasets download -d mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images
  ```

Recommended practice:
- Keep all OOD datasets **outside git**.
- Share download links + expected folder layout in this section.

## Environment setup (local)

From the repo root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Device support

- Local macOS Apple Silicon: automatically uses **MPS** (`torch.device("mps")`) when available.
- Colab: uses **CUDA** when available.
- Otherwise: CPU.

The selection logic lives in `utils.py` (`get_device()`), and scripts print `Using device: ...` at runtime.

---

## Quick sanity check

Verify the data loader, transforms, and split sizes:

```bash
python3 verify_data.py
```

Expected output includes something like:
- `Len train/val/test: 5040 1080 1080`
- `x shape torch.Size([4, 3, 224, 224])`

---

## Training (local or Colab)

### Baseline (EfficientNet-B0 + Adam)

```bash
python3 train.py --model efficientnet_b0 --epochs 30 --batch_size 64 --num_workers 4
```

### SAM (EfficientNet-B0 + SAM)

```bash
python3 train.py --model efficientnet_b0 --use_sam --rho 0.05 --epochs 30 --batch_size 64 --num_workers 4
```

Notes:
- Training prints tqdm progress bars for Train/Val loops.
- Checkpoints save to `outputs/checkpoints/` as:
  - `best_efficientnet_b0_baseline.pt`
  - `best_efficientnet_b0_sam.pt`

If you only want a quick test run:

```bash
python3 train.py --model efficientnet_b0 --epochs 2 --max_batches 50
```

---

## Evaluation (test set)

```bash
python3 eval.py --checkpoint outputs/checkpoints/best_efficientnet_b0_sam.pt --output_dir outputs/results
```

Artifacts:
- `outputs/results/metrics_<checkpoint_stem>.json`
- `outputs/results/metrics_<checkpoint_stem>.md`

Metrics included:
- Accuracy
- Macro-F1
- Macro Precision/Recall
- AUROC (macro, OVR)
- Confusion matrix

---

## Explainability (batch / dataset)

Generate Grad-CAM + Integrated Gradients figures for a few test samples:

```bash
python3 explain.py --checkpoint outputs/checkpoints/best_efficientnet_b0_sam.pt --output_dir outputs/results --num_samples 8
```

Saved to:
- `outputs/results/explainability/`

---

## Single-image inference + Grad-CAM (internet images)

### `inference.py`

Runs prediction + Grad-CAM for a **single image path**, and saves:
- `outputs/results/explainability/latest_inference.jpg`

```bash
python3 inference.py "/absolute/or/relative/path/to/image.jpg"
```

Important:
- Quote paths that contain spaces.
- Uses ImageNet normalization.
- Loads `outputs/checkpoints/best_efficientnet_b0_sam.pt`.

### `pipeline.py`

End-to-end single-image prediction + Grad-CAM for raw “internet” images using:
- `Resize(256)` + `CenterCrop(224)` (safe border removal)
- ImageNet normalization

Saves:
- `outputs/results/explainability/latest_pipeline_result.jpg`

```bash
python3 pipeline.py "/absolute/or/relative/path/to/image.jpg"
```

---

## Colab workflow (what we ran)

Notebook: `Colab/DLCV_Project.ipynb`

High-level steps used:
- Mount Google Drive
- Unzip code zip into `/content/project`
- Unzip data into `/content/project/data` (and fix the “double folder” case)
- Install `requirements.txt`
- Run:
  - `python verify_data.py`
  - `python train.py --model efficientnet_b0 --epochs 30 --batch_size 32 --lr 0.001`
  - `python train.py --model efficientnet_b0 --use_sam --epochs 30 --batch_size 32 --lr 0.001`
  - `python train.py --model resnet50 --epochs 30 --batch_size 32 --lr 0.001`
- Copy `.pt` checkpoints to Google Drive for sharing/backup

If you want to reproduce exactly, open the notebook and run cells top-to-bottom.

---

## Project layout (core files)

- `config.py` — dataset paths and hyperparameters (epochs, lr, rho, etc.)
- `dataset.py` — dataset + deterministic split + dataloaders
- `transforms.py` — **train vs val/test transforms** (heavy augmentation for train, clean val/test)
- `models.py` — EfficientNet-B0 / ResNet-50 heads (4-class)
- `losses.py` — Focal Loss
- `sam.py` — SAM step implementation
- `train.py` — training entrypoint (baseline + SAM) with tqdm progress bars
- `metrics.py` — evaluation metrics
- `eval.py` — test-set evaluation (writes JSON/MD)
- `explainability/gradcam.py` — Grad-CAM
- `explainability/integrated_gradients.py` — Integrated Gradients
- `explain.py` — explainability runner over test samples
- `inference.py` — single image inference + Grad-CAM
- `pipeline.py` — single image pipeline (Resize 256 + CenterCrop 224) + Grad-CAM
- `Execution_Instructions.md` — command-by-command execution notes

---

## Collaboration tips

- **Don’t commit**: `data/`, `outputs/`, `*.pt`, `venv/`, `__pycache__/` (already in `.gitignore`)
- **Share checkpoints**: Google Drive folder / shared cloud drive
- **Reproducibility**: keep hyperparameters in `config.py` or pass explicit CLI flags (epochs, lr, rho, batch size)

---

## References

- EfficientNet (Tan & Le, ICML 2019)
- ResNet (He et al., CVPR 2016)
- Sharpness-Aware Minimization (Foret et al., ICLR 2021)
- Grad-CAM (Selvaraju et al., ICCV 2017)
- Integrated Gradients (Sundararajan et al., ICML 2017)
- Focal Loss (Lin et al., ICCV 2017)
