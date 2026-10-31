from __future__ import annotations

import argparse
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from cxrbench.analysis import ensemble_agreement, validation_selection_table
from cxrbench.config import CLASSES, load_config
from cxrbench.data import (
    duplicate_cross_split_report,
    duplicate_group_summary,
    find_images,
    hash_images,
    split_clean,
)
from cxrbench.evaluation import build_results_table, classification_details, per_class_recall
from cxrbench.protocols import PAPER_CI, PAPER_CM, PAPER_K, PAPER_N
from cxrbench.training import result_path
from cxrbench.utils import json_safe_records, save_json, short_name
from cxrbench.visualize import plot_confusion_matrices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cached result files and build the comparison table."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="A")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {"data_dir": args.data_dir, "out_dir": args.out_dir}
    cfg = load_config(args.config, **overrides)
    epochs = args.epochs if args.epochs is not None else cfg.epochs

    df = find_images(cfg.data_dir)
    if df.empty:
        raise SystemExit(
            f"No images found under {cfg.data_dir!r}. See data/README.md for the expected layout."
        )
    labels = df["y"].to_numpy()
    items_a = split_clean(labels, cfg.split_seed)

    hashed = hash_images(df)
    group_summary = duplicate_group_summary(hashed)
    duplicate_audits = [
        duplicate_cross_split_report(hashed, split_seed)
        for split_seed in cfg.duplicate_split_seeds
    ]
    print(
        f"Exact duplicate groups: {group_summary['groups']} | files: "
        f"{group_summary['files']} | conflicting labels: "
        f"{group_summary['conflicting_label_groups']}"
    )
    for audit in duplicate_audits:
        print(
            f"  split {audit['split_seed']:>2}: spanning >1 subset = "
            f"{audit['groups_spanning_subsets']} | train&test = "
            f"{audit['train_test_groups']} | test with train duplicate = "
            f"{audit['test_images_with_train_duplicate']}"
        )

    results = {}
    missing = []
    for model_name in cfg.models:
        for seed in cfg.seeds:
            path = result_path(cfg, args.tag, model_name, seed, epochs)
            if not path.exists():
                missing.append(str(path))
                continue
            with open(path, "rb") as handle:
                results[(model_name, seed)] = pickle.load(handle)
    if missing:
        raise SystemExit(
            "Missing result files:\n" + "\n".join(missing) +
            "\nRun train.py first or check --out-dir/--tag/--epochs."
        )

    table, ensemble_probs = build_results_table(cfg, results, items_a[2], labels)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print("=== Protocol A results ===")
    print(table.round(4).to_string(index=False))
    print(
        "\np_vs_paper: one-sided exact binomial test, H0: our true accuracy <= "
        f"{PAPER_K}/{PAPER_N}. Treats the published number as fixed; the comparison "
        "is indicative because the original test images are not public."
    )
    table.to_csv(cfg.out_path / "results_table.csv", index=False)

    y_true = labels[items_a[2][:, 0]]

    validation_table = validation_selection_table(
        cfg, results, items_a[1], labels
    )
    print("\n=== Architecture selection on the validation split ===")
    print(validation_table.round(2).to_string(index=False))

    agreement = ensemble_agreement(
        ensemble_probs, y_true, short_name(cfg.models[0])
    )
    print(
        "\nEnsemble agreement: identical predictions on "
        f"{agreement['identical_fraction']:.4f} of test images | "
        f"disagreements {agreement['disagreements']} | "
        f"correct in both {agreement['correct_in_both']} | "
        f"only reference correct {agreement['only_reference_correct']} | "
        f"only combined correct {agreement['only_combined_correct']}"
    )

    best_record = max(table.iloc[:-1].to_dict("records"), key=lambda row: row["acc"])
    best_label = best_record["model"]
    best_key = best_label.split(" ")[0]
    best_probs = ensemble_probs.get(
        best_key, np.mean(list(ensemble_probs.values()), axis=0)
    )

    report, confusion, predictions = classification_details(y_true, best_probs)
    print(f"\nBest configuration: {best_label}")
    print(classification_report(y_true, predictions, target_names=CLASSES, digits=3))
    print(
        pd.DataFrame(
            {"recall": per_class_recall(confusion)},
            index=CLASSES,
        ).round(3)
    )

    fig = plot_confusion_matrices(
        confusion,
        PAPER_CM,
        CLASSES,
        best_label,
        cfg.out_path / "confusion_matrices.png",
    )
    plt.close(fig)

    save_json(
        cfg.out_path / "evaluation.json",
        {
            "paper": {
                "correct": f"{PAPER_K}/{PAPER_N}",
                "ci95": PAPER_CI,
            },
            "best_model": best_label,
            "classification_report": report,
            "confusion_matrix": confusion.tolist(),
            "results": json_safe_records(table.drop(columns=["note"])),
            "validation_selection": json_safe_records(validation_table),
            "ensemble_agreement": agreement,
            "duplicate_groups": group_summary,
            "duplicate_audits": duplicate_audits,
        },
    )
    print(f"\nSaved evaluation to {cfg.out_path}")


if __name__ == "__main__":
    main()
