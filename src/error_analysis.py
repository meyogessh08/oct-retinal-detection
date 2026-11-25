"""Error analysis on the test split for MobileNetV2."""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset

from src import config
from src.models import load_checkpoint
from src.preprocessing import get_class_names, get_transforms


def load_test_split(dataset_id: str = config.HF_DATASET_ID):
    """Download/load ONLY the test split so the ~6 GB train split isn't pulled."""
    return load_dataset(dataset_id, split="test")


def main():
    ckpt_path = config.MODEL_DIR / "mobilenet_v2_fine_tune.pt"
    misclass_path = config.RESULT_DIR / "mobilenet_v2_fine_tune_misclassified.npy"
    out_path = config.FIGURE_DIR / "mobilenet_v2_misclassified_grid.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, class_names = load_checkpoint(ckpt_path, device=device)

    test_ds = load_test_split()
    try:
        ds_names = list(test_ds.features["label"].names)
        if ds_names:
            class_names = ds_names
    except Exception:
        pass

    mis_idx = np.load(misclass_path).tolist()
    transform = get_transforms(train=False, pretrained=True)

    rows_data = []
    drusen_to_cnv = []

    with torch.no_grad():
        for idx in mis_idx:
            row = test_ds[int(idx)]
            pil_img = row["image"].convert("RGB")
            true_id = int(row["label"])

            tensor = transform(pil_img).unsqueeze(0).to(device)
            probs = torch.softmax(model(tensor), dim=1)[0]
            pred_id = int(probs.argmax().item())
            conf = float(probs[pred_id].item())

            true_name = class_names[true_id]
            pred_name = class_names[pred_id]
            rows_data.append((idx, pil_img, true_name, pred_name, conf))

            if true_name == "DRUSEN" and pred_name == "CNV":
                drusen_to_cnv.append(int(idx))

    print(f"Loaded {len(test_ds)} test images | Total misclassified: {len(rows_data)}")
    print(f"DRUSEN -> CNV indices ({len(drusen_to_cnv)}): {drusen_to_cnv}")

    if not rows_data:
        print("No misclassified samples found; skipping grid plot.")
        return

    cols = 5
    n_imgs = len(rows_data)
    n_rows = math.ceil(n_imgs / cols)
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 3.2, n_rows * 3.0))
    axes = np.atleast_1d(axes).ravel()

    for ax, (idx, img, true_name, pred_name, conf) in zip(axes, rows_data):
        ax.imshow(img)
        ax.set_title(f"#{idx} T:{true_name}\nP:{pred_name} ({conf:.2f})", fontsize=8)
        ax.axis("off")

    for ax in axes[n_imgs:]:
        ax.axis("off")

    fig.suptitle("MobileNetV2 Misclassified Test Samples", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved misclassified grid to {Path(out_path)}")


if __name__ == "__main__":
    main()