# Execution Instructions — Brain Tumor Classification with SAM

Step-by-step guide to run the project from a clean state.

---

## 1. Set up the environment

From the project root:

```bash
cd "/Users/srinathganesh/Personal/UIC/Semester 4/Deep Learning in Computer Vision/Project Implementation"

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## 2. Data

You should already have the `data/` folder. The expected layout is:

- `data/Training/glioma`, `data/Training/meningioma`, `data/Training/notumor`, `data/Training/pituitary`
- `data/Testing/` with the same four class folders

If you need to download the dataset from Kaggle:

```bash
mkdir -p data
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data
cd data && unzip brain-tumor-mri-dataset.zip && cd ..
```

---

## 3. Verify the data pipeline

```bash
python3 verify_data.py
```

You should see train/val/test sizes (e.g. 5040 / 1080 / 1080), batch shape `(4, 3, 224, 224)`, and **"Data pipeline OK."**

---

## 4. Train the model

**Baseline (no SAM):**

```bash
python3 train.py --model efficientnet_b0 --epochs 30
```

Checkpoints are saved under `outputs/checkpoints/`, e.g.  
`outputs/checkpoints/best_efficientnet_b0_baseline.pt`.

**With SAM:**

```bash
python3 train.py --model efficientnet_b0 --use_sam --rho 0.05 --epochs 30
```

Checkpoint will be named like `outputs/checkpoints/best_efficientnet_b0_sam.pt`.

**Quick test run (few batches):**

```bash
python3 train.py --model efficientnet_b0 --epochs 2 --max_batches 50
```

---

## 5. Evaluate on the test set

Use the checkpoint that matches how you trained.

**If you ran baseline:**

```bash
python3 eval.py --checkpoint outputs/checkpoints/best_efficientnet_b0_baseline.pt --output_dir outputs/results
```

**If you ran SAM:**

```bash
python3 eval.py --checkpoint outputs/checkpoints/best_efficientnet_b0_sam.pt --output_dir outputs/results
```

This writes:

- `outputs/results/metrics_<checkpoint_stem>.json`
- `outputs/results/metrics_<checkpoint_stem>.md`

and prints accuracy, Macro-F1, precision, recall, AUROC, and confusion matrix.

---

## 6. Run explainability (Grad-CAM + Integrated Gradients)

Use the same checkpoint as in step 5.

**Baseline:**

```bash
python3 explain.py --checkpoint outputs/checkpoints/best_efficientnet_b0_baseline.pt --output_dir outputs/results
```

**SAM:**

```bash
python3 explain.py --checkpoint outputs/checkpoints/best_efficientnet_b0_sam.pt --output_dir outputs/results
```

Figures are saved under `outputs/results/explainability/` (input, Grad-CAM, IG, overlay).

**Fewer samples (e.g. 4):**

```bash
python3 explain.py --checkpoint outputs/checkpoints/best_efficientnet_b0_baseline.pt --num_samples 4
```

---

## 7. Optional: ResNet-50

Same flow with a different model:

```bash
python3 train.py --model resnet50 --epochs 30
python3 eval.py --checkpoint outputs/checkpoints/best_resnet50_baseline.pt --output_dir outputs/results
python3 explain.py --checkpoint outputs/checkpoints/best_resnet50_baseline.pt --output_dir outputs/results
```

---

## Quick reference

| Step | Command |
|------|--------|
| 1. Environment | `python3 -m venv venv` → `source venv/bin/activate` → `pip install -r requirements.txt` |
| 2. Data | Already present, or Kaggle download as above |
| 3. Verify | `python3 verify_data.py` |
| 4. Train baseline | `python3 train.py --model efficientnet_b0 --epochs 30` |
| 5. Evaluate | `python3 eval.py --checkpoint outputs/checkpoints/best_efficientnet_b0_baseline.pt --output_dir outputs/results` |
| 6. Explain | `python3 explain.py --checkpoint outputs/checkpoints/best_efficientnet_b0_baseline.pt --output_dir outputs/results` |

---

## Configuration

Paths, batch size, learning rate, SAM ρ, Focal Loss γ, etc. are in **`config.py`**. Edit that file to change hyperparameters or data paths.
