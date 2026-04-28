"""Monitoring dashboard for training runs.

Run with:
    streamlit run dashboard.py
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import streamlit as st

import config
from monitoring import read_jsonl


st.set_page_config(page_title="NeuroScan Monitoring Dashboard", layout="wide")

st.title("NeuroScan Monitoring Dashboard")
st.caption("Training curves, validation metrics, and Grad-CAM snapshots across epochs.")

with st.sidebar:
    st.header("Data sources")
    history_file = Path(st.text_input("History JSONL", value=str(config.HISTORY_PATH)))
    gradcam_dir = Path(st.text_input("Grad-CAM folder", value=str(config.GRADCAM_DIR)))
    if st.button("Refresh"):
        st.rerun()

records: List[Dict] = read_jsonl(history_file)

if not records:
    st.warning(f"No monitoring history found at {history_file}. Run training first to generate logs.")
    st.stop()

latest = records[-1]
best = max(records, key=lambda r: r.get("macro_f1", -1.0))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest epoch", latest.get("epoch", "-"))
m2.metric("Latest val Macro-F1", f"{latest.get('macro_f1', float('nan')):.4f}")
m3.metric("Latest val accuracy", f"{latest.get('accuracy', float('nan')):.4f}")
m4.metric("Best val Macro-F1", f"{best.get('macro_f1', float('nan')):.4f}")

cols = st.columns(2)
with cols[0]:
    st.subheader("Training curves")
    epochs = [r.get("epoch") for r in records]
    train_loss = [r.get("train_loss") for r in records]
    val_f1 = [r.get("macro_f1") for r in records]
    val_acc = [r.get("accuracy") for r in records]
    val_auroc = [r.get("auroc") for r in records]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, train_loss, label="train_loss")
    ax.plot(epochs, val_f1, label="val_macro_f1")
    ax.plot(epochs, val_acc, label="val_accuracy")
    if any(v is not None for v in val_auroc):
        ax.plot(epochs, val_auroc, label="val_auroc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Loss / validation metrics")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)

with cols[1]:
    st.subheader("Grad-CAM monitoring")
    gradcam_rows = [r for r in records if r.get("gradcam_mean") is not None]
    gradcam_epochs = [r.get("epoch") for r in gradcam_rows]
    gradcam_mean = [r.get("gradcam_mean") for r in gradcam_rows]
    gradcam_cov = [r.get("gradcam_coverage") for r in gradcam_rows]
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    if gradcam_mean:
        ax2.plot(gradcam_epochs, gradcam_mean, label="gradcam_mean")
    if gradcam_cov:
        ax2.plot(gradcam_epochs, gradcam_cov, label="gradcam_coverage")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Grad-CAM signal across epochs")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    st.pyplot(fig2, clear_figure=True)

st.subheader("Latest epoch snapshot")
preview = latest.get("gradcam_preview")
if preview:
    preview_path = Path(preview)
    if not preview_path.is_absolute():
        preview_path = Path.cwd() / preview_path
    if preview_path.exists():
        st.image(str(preview_path), caption=f"Epoch {latest.get('epoch')} Grad-CAM snapshot", use_container_width=True)
    else:
        st.info(f"Grad-CAM preview not found: {preview_path}")
else:
    st.info("No Grad-CAM preview recorded for the latest epoch.")

st.subheader("Recent records")
st.json(records[-5:])

st.caption(f"Monitoring files: {history_file}  |  {gradcam_dir}")
