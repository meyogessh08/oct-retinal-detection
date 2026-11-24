"""Dataset exploration (brief section 1-2): class counts, image sizes, sample grid.

Run:  python -m src.explore
"""
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.preprocessing import class_counts, get_class_names, load_oct_dataset


def plot_class_distribution(counts_train, counts_test, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, counts, title in zip(axes, (counts_train, counts_test), ("train", "test")):
        vals = [counts.get(i, 0) for i in range(len(class_names))]
        ax.bar(class_names, vals, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
        ax.set_title(f"{title} class distribution")
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_samples(split, class_names, per_class=4):
    labels = np.asarray(split["label"])
    rng = np.random.default_rng(config.SEED)
    fig, axes = plt.subplots(len(class_names), per_class, figsize=(3 * per_class, 3 * len(class_names)))
    for r, name in enumerate(class_names):
        for c, idx in enumerate(rng.choice(np.where(labels == r)[0], per_class, replace=False)):
            img = split[int(idx)]["image"]
            axes[r, c].imshow(img, cmap="gray")
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_title(name, loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "sample_images.png", dpi=120)
    plt.close(fig)


def main():
    ds = load_oct_dataset()
    class_names = get_class_names(ds)
    print("Class names:", class_names)

    ctr, cte = class_counts(ds["train"]), class_counts(ds["test"])
    print("Train counts:", ctr)
    print("Test counts: ", cte)
    plot_class_distribution(ctr, cte, class_names)

    sizes = [ds["train"][i]["image"].size for i in range(0, len(ds["train"]), 5000)]
    modes = {ds["train"][i]["image"].mode for i in range(0, len(ds["train"]), 5000)}
    print("Sample image sizes (w, h):", sorted(set(sizes))[:10], "| modes:", modes)

    plot_samples(ds["train"], class_names)
    print("Saved figures to outputs/figures/")


if __name__ == "__main__":
    main()
