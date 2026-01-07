#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_true_pred(csv_path: Path):
    """Read true / pred labels from predictions.csv produced by eval_ipn.py."""
    y_true = []
    y_pred = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # columns: row,true,pred,top1,top2,top3,top4,top5,p1,p2,p3,p4,p5
        for row in reader:
            y_true.append(int(row["true"]))
            y_pred.append(int(row["pred"]))
    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int)


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray, num_class: int):
    """Standard (non-normalized) confusion matrix [num_class, num_class]."""
    cm = np.zeros((num_class, num_class), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_rownorm(
    cm: np.ndarray,
    out_png: Path,
    title: str = "Confusion Matrix",
):
    """
    Plot a row-normalized confusion matrix:
    - Colors are based on cm_row_norm in [0,1]
    - Each non-zero cell shows: 'xx.x%\\n(count)'
    """
    num_class = cm.shape[0]

    # Row-normalize for coloring
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums  # values in [0,1] per row

    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-normalized")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    ax.set_xticks(np.arange(num_class))
    ax.set_yticks(np.arange(num_class))

    # Annotate each non-zero cell with percentage (row-wise) and count
    for i in range(num_class):
        for j in range(num_class):
            count = cm[i, j]
            if count == 0:
                continue
            pct = 100.0 * cm_norm[i, j]
            # text = f"{pct:.1f}%\n({count})"
            text = f"{pct:.1f}%"

            # Choose text color for readability
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred-csv",
        type=str,
        required=True,
        help="Path to predictions.csv produced by eval_ipn.py --dump-preds",
    )
    ap.add_argument(
        "--num-class",
        type=int,
        required=True,
        help="Number of IPN classes (e.g., 24)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to save confusion_matrix.* (default: same as csv)",
    )
    args = ap.parse_args()

    csv_path = Path(args.pred_csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = load_true_pred(csv_path)
    num_class = args.num_class

    cm = compute_confusion(y_true, y_pred, num_class)

    # Save raw matrix
    np.save(out_dir / "confusion_matrix.npy", cm)
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    # Plot row-normalized matrix with (% and count)
    plot_confusion_rownorm(cm, out_dir / "confusion_matrix_rownorm.pdf")
    plot_confusion_rownorm(cm, out_dir / "confusion_matrix_rownorm.png")
    plot_confusion_rownorm(cm, out_dir / "confusion_matrix_rownorm.svg")
    print(f"Saved confusion matrix (raw + plot) to: {out_dir}")


if __name__ == "__main__":
    main()
