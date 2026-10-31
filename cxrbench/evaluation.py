from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .config import CLASSES, Config
from .protocols import PAPER_ACC, PAPER_CI, PAPER_K, PAPER_N
from .utils import bootstrap_ci, short_name


def result_row(
    label: str,
    probabilities,
    y_true: np.ndarray,
    extra: str = "",
) -> dict:
    if isinstance(probabilities, (list, tuple)):
        probs = np.mean(probabilities, axis=0)
    else:
        probs = probabilities
    predictions = probs.argmax(1)
    correct = int((predictions == y_true).sum())
    n_samples = len(y_true)
    ci_low, ci_high = bootstrap_ci(y_true, predictions)
    p_value = stats.binomtest(
        correct, n_samples, PAPER_ACC, alternative="greater"
    ).pvalue
    return {
        "model": label,
        "acc": correct / n_samples,
        "ci_lo": ci_low,
        "ci_hi": ci_high,
        "macroF1": f1_score(y_true, predictions, average="macro"),
        "correct": f"{correct}/{n_samples}",
        "p_vs_paper": p_value,
        "note": extra,
    }


def build_results_table(
    cfg: Config,
    results: Mapping[tuple[str, int], dict],
    test_items: np.ndarray,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    y_true = labels[test_items[:, 0]]
    rows = []
    ensemble_probs: dict[str, np.ndarray] = {}

    for model_name in cfg.models:
        key = short_name(model_name)
        accuracies = [
            float(
                (
                    results[(model_name, seed)]["probs_te"].argmax(1)
                    == y_true
                ).mean()
            )
            for seed in cfg.seeds
        ]
        rows.append(
            result_row(
                f"{key} (seed-ensemble)",
                [results[(model_name, seed)]["probs_te"] for seed in cfg.seeds],
                y_true,
                extra=(
                    f"single-run mean±std = {np.mean(accuracies) * 100:.2f}"
                    f"±{(np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0) * 100:.2f}"
                ),
            )
        )
        ensemble_probs[key] = np.mean(
            [results[(model_name, seed)]["probs_te"] for seed in cfg.seeds], axis=0
        )

    rows.append(
        result_row(
            "ALL-MODEL ENSEMBLE",
            list(ensemble_probs.values()),
            y_true,
            extra="average of seed-ensembles",
        )
    )

    table = pd.DataFrame(rows)
    table.loc[len(table)] = {
        "model": "PAPER ATL-ResNeXt50 (non-aug)",
        "acc": PAPER_ACC,
        "ci_lo": PAPER_CI[0],
        "ci_hi": PAPER_CI[1],
        "macroF1": np.nan,
        "correct": f"{PAPER_K}/{PAPER_N}",
        "p_vs_paper": np.nan,
        "note": "reported non-augmented test result",
    }
    return table, ensemble_probs


def classification_details(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[dict, np.ndarray, np.ndarray]:
    predictions = probabilities.argmax(1)
    report = classification_report(
        y_true, predictions, target_names=CLASSES, digits=3, output_dict=True
    )
    matrix = confusion_matrix(y_true, predictions)
    return report, matrix, predictions


def per_class_recall(matrix: np.ndarray) -> np.ndarray:
    return matrix.diagonal() / matrix.sum(1)
