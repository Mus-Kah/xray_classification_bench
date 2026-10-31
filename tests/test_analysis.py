import numpy as np
import pandas as pd

from cxrbench.analysis import (
    accuracy_breakdown,
    ensemble_agreement,
    validation_selection_table,
)
from cxrbench.config import Config
from cxrbench.data import duplicate_cross_split_report, duplicate_group_summary


def _one_hot(predictions, n_classes=3):
    return np.eye(n_classes)[predictions]


def test_accuracy_breakdown_separates_originals_and_augmented():
    y_true = np.array([0, 1, 0, 1])
    variants = np.array([0, 0, 1, 1])
    predictions = np.array([0, 0, 0, 1])
    breakdown = accuracy_breakdown(_one_hot(predictions), y_true, variants)
    assert breakdown["overall"] == 75.0
    assert breakdown["original"] == 50.0
    assert breakdown["augmented"] == 100.0


def test_ensemble_agreement_counts_disagreements():
    y_true = np.array([0, 1, 2, 0])
    ensemble_probs = {
        "reference": _one_hot([0, 1, 2, 0]),
        "other_b": _one_hot([1, 1, 2, 0]),
        "other_c": _one_hot([1, 1, 2, 0]),
    }
    agreement = ensemble_agreement(ensemble_probs, y_true, "reference")
    assert agreement["identical_fraction"] == 0.75
    assert agreement["disagreements"] == 1
    assert agreement["correct_in_both"] == 3
    assert agreement["only_reference_correct"] == 1
    assert agreement["only_combined_correct"] == 0


def test_validation_selection_uses_validation_probabilities():
    cfg = Config(models=["m"], seeds=[0], quick_test=False)
    val_items = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    labels = np.array([0, 1, 2, 0])
    correct = _one_hot([0, 1, 2, 0])
    results = {("m", 0): {"probs_va": correct}}
    table = validation_selection_table(cfg, results, val_items, labels)
    assert table.loc[0, "val_seed_ensemble"] == 100.0
    assert table.loc[0, "n_val"] == 4


def test_duplicate_cross_split_report_counts_test_duplicates():
    hashed = pd.DataFrame(
        {"y": np.zeros(300, dtype=int), "h": ["same"] * 300}
    )
    summary = duplicate_group_summary(hashed)
    assert summary["groups"] == 1
    assert summary["files"] == 300
    audit = duplicate_cross_split_report(hashed, split_seed=42)
    assert audit["groups_spanning_subsets"] == 1
    assert audit["train_test_groups"] == 1
    assert audit["test_images_with_train_duplicate"] == 90
