from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_class_samples(raw, clahe, labels, classes, path: str | Path | None = None):
    fig, axes = plt.subplots(2, len(classes), figsize=(3 * len(classes), 6))
    for index, class_name in enumerate(classes):
        sample = np.where(labels == index)[0][0]
        axes[0, index].imshow(raw[sample])
        axes[0, index].set_title(class_name)
        axes[0, index].axis("off")
        axes[1, index].imshow(clahe[sample])
        axes[1, index].axis("off")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrices(
    confusion: np.ndarray,
    paper_confusion: np.ndarray,
    classes,
    ours_label: str,
    path: str | Path | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, matrix, title in zip(
        axes,
        [confusion, paper_confusion],
        [f"Ours ({ours_label})", "Published baseline"],
    ):
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(title, fontsize=9)
        axis.set_xticks(range(len(classes)))
        axis.set_xticklabels(classes, rotation=20, fontsize=8)
        axis.set_yticks(range(len(classes)))
        axis.set_yticklabels(classes, fontsize=8)
        for row in range(len(classes)):
            for col in range(len(classes)):
                axis.text(
                    col,
                    row,
                    matrix[row, col],
                    ha="center",
                    va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black",
                )
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
