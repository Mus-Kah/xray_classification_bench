from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .utils import short_name


def _std(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def accuracy_breakdown(
    probabilities: np.ndarray, y_true: np.ndarray, variants: np.ndarray
) -> dict:
    """Accuracy overall and split by unaugmented originals vs offline variants."""
    correct = probabilities.argmax(1) == y_true
    originals = np.asarray(variants) == 0
    return {
        "overall": float(correct.mean() * 100),
        "original": float(correct[originals].mean() * 100),
        "augmented": float(correct[~originals].mean() * 100),
    }


def protocol_breakdown(
    results: Mapping[int, dict],
    items: tuple[np.ndarray, np.ndarray, np.ndarray],
    labels: np.ndarray,
    seeds: Sequence[int],
) -> dict:
    """Per-seed accuracy breakdown for one augment-before-split style protocol."""
    y_true = labels[items[2][:, 0]]
    variants = items[2][:, 1]
    rows = [accuracy_breakdown(results[seed]["probs_te"], y_true, variants) for seed in seeds]
    summary = {key: [row[key] for row in rows] for key in rows[0]}
    return {
        "runs": summary,
        "mean": {key: float(np.mean(values)) for key, values in summary.items()},
        "std": {key: _std(values) for key, values in summary.items()},
    }


def protocol_b_prime_comparison(
    protocol_a_accuracy: float,
    protocol_b: dict,
    protocol_b_prime: dict,
) -> dict:
    """Contrast Protocol B (sibling overlap) with Protocol B' (no overlap)."""
    b_mean = protocol_b["mean"]
    bp_mean = protocol_b_prime["mean"]
    return {
        "protocol_a_accuracy": float(protocol_a_accuracy),
        "protocol_b_accuracy": b_mean["overall"] / 100,
        "protocol_b_prime_accuracy": bp_mean["overall"] / 100,
        "leakage_effect_points": b_mean["overall"] - bp_mean["overall"],
        "leakage_effect_originals_points": b_mean["original"]
        - bp_mean["original"],
        "budget_test_composition_effect_points": bp_mean["overall"]
        - protocol_a_accuracy * 100,
        "protocol_b": protocol_b,
        "protocol_b_prime": protocol_b_prime,
    }


def validation_selection_table(
    cfg: Config,
    results: Mapping[tuple[str, int], dict],
    val_items: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Compare architectures on the validation split, never on the test split."""
    y_val = labels[val_items[:, 0]]
    rows = []
    for model_name in cfg.models:
        probabilities = [
            results[(model_name, seed)]["probs_va"] for seed in cfg.seeds
        ]
        accuracies = [
            float((probs.argmax(1) == y_val).mean() * 100)
            for probs in probabilities
        ]
        ensemble_accuracy = float(
            (np.mean(probabilities, axis=0).argmax(1) == y_val).mean() * 100
        )
        rows.append(
            {
                "model": short_name(model_name),
                "val_single_run_mean": float(np.mean(accuracies)),
                "val_single_run_std": _std(accuracies),
                "val_seed_ensemble": ensemble_accuracy,
                "n_val": len(y_val),
            }
        )
    return pd.DataFrame(rows)


def ensemble_agreement(
    ensemble_probs: Mapping[str, np.ndarray],
    y_true: np.ndarray,
    reference: str,
) -> dict:
    """Compare one model ensemble with the average of all model ensembles."""
    reference_predictions = ensemble_probs[reference].argmax(1)
    combined_predictions = np.mean(list(ensemble_probs.values()), axis=0).argmax(1)
    reference_correct = reference_predictions == y_true
    combined_correct = combined_predictions == y_true
    return {
        "reference": reference,
        "identical_fraction": float(
            (reference_predictions == combined_predictions).mean()
        ),
        "disagreements": int((reference_predictions != combined_predictions).sum()),
        "correct_in_both": int((reference_correct & combined_correct).sum()),
        "only_reference_correct": int(
            (reference_correct & ~combined_correct).sum()
        ),
        "only_combined_correct": int(
            (~reference_correct & combined_correct).sum()
        ),
    }
