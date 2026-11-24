"""Evaluation on the held-out test set: metrics, confusion matrix, misclassifications,
inference time, and a comparison table across models.

Usage:
    python -m src.evaluate --checkpoints outputs/models/*.pt
"""
import argparse
import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from src import config
from src.models import load_checkpoint
from src.preprocessing import get_dataloaders, load_oct_dataset
from src.utils import get_device, save_json


@torch.no_grad()
def predict(model, loader, device):
    """Return (true_labels, predicted_labels, seconds_per_image)."""
    model.eval().to(device)
    y_true, y_pred = [], []
    start, n = time.time(), 0
    for images, labels in loader:
        out = model(images.to(device))
        y_pred.extend(out.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())
        n += images.size(0)
    return np.array(y_true), np.array(y_pred), (time.time() - start) / n


def evaluate_checkpoint(ckpt_path, dataset, device):
    """Evaluate one saved model on the test set and save its confusion matrix."""
    model, model_name, class_names = load_checkpoint(ckpt_path, device)
    _, _, test_loader = get_dataloaders(dataset, pretrained=model_name != "custom_cnn")
    y_true, y_pred, sec_per_img = predict(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                       zero_division=0)
    run_name = str(ckpt_path).split("/")[-1].replace(".pt", "")

    # confusion matrix figure
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues",
                                                                colorbar=False)
    ax.set_title(f"{run_name}\nacc={acc:.3f}")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / f"{run_name}_confusion.png", dpi=150)
    plt.close(fig)

    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    save_json(report, config.RESULT_DIR / f"{run_name}_report.json")
    print(f"\n=== {run_name} ===")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    wrong = np.where(y_true != y_pred)[0]  # indices into the test set, for error analysis
    np.save(config.RESULT_DIR / f"{run_name}_misclassified.npy", wrong)

    size_mb = ckpt_path_size_mb(ckpt_path)
    return {"model": run_name, "accuracy": acc, "precision_macro": prec,
            "recall_macro": rec, "f1_macro": f1,
            "inference_ms_per_image": sec_per_img * 1000, "model_size_mb": size_mb}


def ckpt_path_size_mb(path) -> float:
    import os
    return os.path.getsize(path) / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="paths to .pt files (default: all in outputs/models)")
    args = parser.parse_args()
    paths = args.checkpoints or sorted(glob.glob(str(config.MODEL_DIR / "*.pt")))

    device = get_device()
    dataset = load_oct_dataset()
    rows = [evaluate_checkpoint(p, dataset, device) for p in paths]

    table = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    table.to_csv(config.RESULT_DIR / "model_comparison.csv", index=False)
    print("\n", table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
