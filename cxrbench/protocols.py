from __future__ import annotations

import numpy as np
from scipy import stats

# Published confusion matrix in the order used by the original figure:
# rows = true Lung Opacity / Normal / Viral Pneumonia.
_PAPER_CM_RAW = np.array([[291, 32, 0], [19, 359, 4], [5, 5, 328]])

# Reorder to the class order used in this project: Normal, Lung Opacity, Viral Pneumonia.
PAPER_CM = _PAPER_CM_RAW[[1, 0, 2]][:, [1, 0, 2]]
PAPER_K = int(np.trace(PAPER_CM))
PAPER_N = int(PAPER_CM.sum())
PAPER_ACC = PAPER_K / PAPER_N
PAPER_CI = tuple(
    float(value)
    for value in stats.binomtest(PAPER_K, PAPER_N).proportion_ci(method="exact")
)


def paper_summary() -> dict:
    return {
        "correct": f"{PAPER_K}/{PAPER_N}",
        "accuracy": PAPER_ACC,
        "ci95": PAPER_CI,
    }
