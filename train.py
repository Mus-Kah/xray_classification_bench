from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from cxrbench.analysis import (
    ensemble_agreement,
    protocol_b_prime_comparison,
    protocol_breakdown,
    validation_selection_table,
)
from cxrbench.config import CLASSES, load_config
from cxrbench.data import (
    duplicate_cross_split_report,
    duplicate_group_summary,
    find_images,
    hash_images,
    load_images,
    sibling_overlap,
    source_overlap,
    split_augment_then_split,
    split_clean,
    split_then_augment,
)
from cxrbench.evaluation import build_results_table
from cxrbench.protocols import PAPER_ACC, PAPER_CI, PAPER_CM, PAPER_K, PAPER_N
from cxrbench.training import train_one
from cxrbench.utils import (
    describe_device,
    json_safe_records,
    resolve_device,
    save_json,
    short_name,
)
from cxrbench.visualize import plot_class_samples, plot_confusion_matrices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a leakage-aware 3-class chest X-ray benchmark."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--no-protocol-b", action="store_true")
    parser.add_argument("--no-protocol-b-prime", action="store_true")
    parser.add_argument(
        "--cpu", action="store_true", help="Allow CPU execution (debugging only)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "data_dir": args.data_dir,
        "out_dir": args.out_dir,
        "quick_test": True if args.quick_test else None,
        "models": args.models,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "run_ablation": False if args.no_ablation else None,
        "run_protocol_b": False if args.no_protocol_b else None,
        "run_protocol_b_prime": False if args.no_protocol_b_prime else None,
    }
    cfg = load_config(args.config, **overrides)
    device = resolve_device(require_cuda=not args.cpu)
    torch.backends.cudnn.benchmark = True

    print(f"Device: {device} | {describe_device(device)}")
    print(f"Torch: {torch.__version__} | CUDA: {torch.version.cuda}")
    print(f"Output directory: {cfg.out_dir}")
    if cfg.quick_test:
        print(">>> QUICK_TEST is ON: results are not final.")
    cfg.out_path.mkdir(parents=True, exist_ok=True)

    df = find_images(cfg.data_dir)
    if df.empty:
        raise SystemExit(
            f"No images found under {cfg.data_dir!r}. See data/README.md for the expected layout."
        )
    print(f"Images: {len(df)}")
    print(df["y"].map(dict(enumerate(CLASSES))).value_counts())
    if len(df) != 3475:
        print(
            f"WARNING: expected 3475 images, found {len(df)}. "
            "Split sizes will fall back to proportions."
        )

    hashed = hash_images(df)
    duplicates = hashed[hashed.duplicated("h", keep=False)]
    group_summary = duplicate_group_summary(hashed)
    print(
        f"Exact duplicate files: {len(duplicates)} | groups: {group_summary['groups']} | "
        "groups with conflicting labels: "
        f"{group_summary['conflicting_label_groups']}"
    )
    duplicate_audits = []
    for split_seed in cfg.duplicate_split_seeds:
        audit = duplicate_cross_split_report(hashed, split_seed)
        duplicate_audits.append(audit)
        print(
            f"Duplicate audit split {split_seed:>2}: groups spanning >1 subset = "
            f"{audit['groups_spanning_subsets']} | train&test groups = "
            f"{audit['train_test_groups']} | test images with a train duplicate = "
            f"{audit['test_images_with_train_duplicate']}"
        )
    labels = df["y"].to_numpy()

    raw_images, clahe_images = load_images(
        df, cfg.image_size, cfg.out_path / f"X_{cfg.image_size}.npz"
    )
    print(f"Image tensors: raw {raw_images.shape}, CLAHE {clahe_images.shape}")
    samples_fig = plot_class_samples(
        raw_images,
        clahe_images,
        labels,
        CLASSES,
        cfg.out_path / "class_samples.png",
    )
    plt.close(samples_fig)

    items_a = split_clean(labels, cfg.split_seed)
    print(
        f"Protocol A sizes: {[len(part) for part in items_a]} | "
        f"test class counts: {np.bincount(labels[items_a[2][:, 0]], minlength=3)}"
    )
    print(
        "Protocol A train/test source-image overlap: "
        f"{source_overlap(items_a[0], items_a[2])}"
    )

    items_b = None
    if cfg.run_protocol_b:
        items_b = split_augment_then_split(labels, cfg.split_seed)
        print(f"Protocol B sizes: {[len(part) for part in items_b]}")
        print(
            "Protocol B test items with a sibling in train: "
            f"{sibling_overlap(items_b[0], items_b[2]):.1%}"
        )

    items_bp = None
    if cfg.run_protocol_b_prime:
        items_bp = split_then_augment(labels, cfg.split_seed)
        print(f"Protocol B' sizes: {[len(part) for part in items_bp]}")
        print(
            "Protocol B' train/test source-image overlap: "
            f"{source_overlap(items_bp[0], items_bp[2])} | "
            "test items with a sibling in train: "
            f"{sibling_overlap(items_bp[0], items_bp[2]):.1%}"
        )

    results_a = {}
    started = time.time()
    for model_name in cfg.models:
        for seed in cfg.seeds:
            checkpoint = None
            if seed == cfg.seeds[0]:
                checkpoint = cfg.out_path / f"ckpt_{short_name(model_name)}_s{seed}.pt"
            results_a[(model_name, seed)] = train_one(
                cfg,
                model_name,
                seed,
                items_a,
                clahe_images,
                labels,
                device,
                epochs=cfg.epochs,
                augment=True,
                tag="A",
                checkpoint=checkpoint,
            )
        print(
            f"--- {model_name}: params "
            f"{results_a[(model_name, cfg.seeds[0])]['n_params']:.1f}M | "
            f"elapsed {(time.time() - started) / 60:.1f} min"
        )

    table, ensemble_probs = build_results_table(
        cfg, results_a, items_a[2], labels
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print("\n=== Protocol A results ===")
    print(table.round(4).to_string(index=False))
    print(
        "\np_vs_paper: one-sided exact binomial test, H0: our true accuracy <= "
        f"{PAPER_K}/{PAPER_N}. Treats the published number as fixed; the comparison "
        "is indicative because the original test images are not public."
    )
    table.to_csv(cfg.out_path / "results_table.csv", index=False)

    y_true = labels[items_a[2][:, 0]]
    best_record = max(
        table.iloc[:-1].to_dict("records"), key=lambda row: row["acc"]
    )
    best_label = best_record["model"]
    best_key = best_label.split(" ")[0]
    best_probs = ensemble_probs.get(
        best_key, np.mean(list(ensemble_probs.values()), axis=0)
    )
    predictions = best_probs.argmax(1)
    print(f"\nBest configuration: {best_label}")
    print(classification_report(y_true, predictions, target_names=CLASSES, digits=3))
    confusion = confusion_matrix(y_true, predictions)
    fig = plot_confusion_matrices(
        confusion,
        PAPER_CM,
        CLASSES,
        best_label,
        cfg.out_path / "confusion_matrices.png",
    )
    plt.close(fig)

    ablation_rows = []
    if cfg.run_ablation:
        print("\n=== Ablation (first model, first three seeds) ===")
        model_name = cfg.models[0]
        seeds = cfg.seeds[:3]
        variants = [
            ("full (CLAHE + on-the-fly aug)", clahe_images, True, None),
            ("no CLAHE (raw pseudo-colour)", raw_images, True, "abl_no_clahe"),
            ("no augmentation", clahe_images, False, "abl_no_aug"),
        ]
        for label, images, augment, tag in variants:
            accuracies = []
            for seed in seeds:
                if tag is None:
                    result = results_a[(model_name, seed)]
                else:
                    result = train_one(
                        cfg,
                        model_name,
                        seed,
                        items_a,
                        images,
                        labels,
                        device,
                        epochs=cfg.epochs,
                        augment=augment,
                        tag=tag,
                    )
                accuracies.append(
                    float((result["probs_te"].argmax(1) == y_true).mean() * 100)
                )
            ablation_rows.append(
                {
                    "variant": label,
                    "mean": float(np.mean(accuracies)),
                    "std": float(
                        np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0
                    ),
                    "runs": np.round(accuracies, 2).tolist(),
                }
            )
        ablation_table = pd.DataFrame(ablation_rows)
        print(ablation_table.round(2).to_string(index=False))
        ablation_table.to_csv(cfg.out_path / "ablation.csv", index=False)

    protocol_b_summary = None
    results_b = {}
    if cfg.run_protocol_b:
        assert items_b is not None
        print("\n=== Protocol B (augment-before-split reference protocol) ===")
        model_name = cfg.models[0]
        y_true_b = labels[items_b[2][:, 0]]
        accuracies_b = []
        for seed in cfg.protocol_b_seeds:
            result = train_one(
                cfg,
                model_name,
                seed,
                items_b,
                clahe_images,
                labels,
                device,
                epochs=cfg.protocol_b_epochs,
                augment=False,
                tag="B",
            )
            results_b[seed] = result
            accuracies_b.append(
                float((result["probs_te"].argmax(1) == y_true_b).mean())
            )
        acc_a = float(
            np.mean(
                [
                    (
                        results_a[(model_name, seed)]["probs_te"].argmax(1)
                        == y_true
                    ).mean()
                    for seed in cfg.seeds
                ]
            )
        )
        protocol_b_summary = {
            "model": short_name(model_name),
            "protocol_a_accuracy": acc_a,
            "protocol_b_accuracy": float(np.mean(accuracies_b)),
            "protocol_b_runs": accuracies_b,
            "delta": float(np.mean(accuracies_b) - acc_a),
            "test_items_with_train_sibling": sibling_overlap(
                items_b[0], items_b[2]
            ),
        }
        print(
            f"{short_name(model_name)} Protocol A: {acc_a * 100:.2f}% | "
            f"Protocol B: {np.mean(accuracies_b) * 100:.2f}%"
        )
        print(
            "Descriptive difference for the same model: "
            f"{protocol_b_summary['delta'] * 100:+.2f} points"
        )

    protocol_b_prime_summary = None
    if cfg.run_protocol_b_prime:
        assert items_bp is not None
        print("\n=== Protocol B' (split-then-augment control) ===")
        model_name = cfg.models[0]
        y_true_bp = labels[items_bp[2][:, 0]]
        accuracies_bp = []
        results_bp = {}
        for seed in cfg.protocol_b_prime_seeds:
            result = train_one(
                cfg,
                model_name,
                seed,
                items_bp,
                clahe_images,
                labels,
                device,
                epochs=cfg.protocol_b_prime_epochs,
                augment=False,
                tag="Bp",
            )
            results_bp[seed] = result
            accuracies_bp.append(
                float((result["probs_te"].argmax(1) == y_true_bp).mean())
            )
        common_seeds = [
            seed for seed in cfg.protocol_b_prime_seeds if seed in results_b
        ]
        if common_seeds:
            breakdown_b = protocol_breakdown(
                results_b, items_b, labels, common_seeds
            )
            breakdown_bp = protocol_breakdown(
                results_bp, items_bp, labels, common_seeds
            )
            protocol_b_prime_summary = protocol_b_prime_comparison(
                acc_a, breakdown_b, breakdown_bp
            )
            protocol_b_prime_summary.update(
                {
                    "model": short_name(model_name),
                    "protocol_b_prime_runs": accuracies_bp,
                    "common_seeds": common_seeds,
                }
            )
            for label, breakdown in (("B", breakdown_b), ("B'", breakdown_bp)):
                for key in ("overall", "original", "augmented"):
                    values = np.round(breakdown["runs"][key], 2)
                    print(
                        f"{label:2s} {key:9s} mean {breakdown['mean'][key]:6.2f} "
                        f"+- {breakdown['std'][key]:.2f}   {values}"
                    )
            print(
                f"Leakage effect B - B': "
                f"{protocol_b_prime_summary['leakage_effect_points']:+.2f} points "
                "| on test originals only: "
                f"{protocol_b_prime_summary['leakage_effect_originals_points']:+.2f}"
            )
            print(
                "Budget/test-composition effect B' - A: "
                f"{protocol_b_prime_summary['budget_test_composition_effect_points']:+.2f} points"
            )
        else:
            print(
                "Protocol B results are unavailable; B' was trained but the "
                "leakage contrast was skipped."
            )

    validation_table = validation_selection_table(
        cfg, results_a, items_a[1], labels
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

    summary = {
        "protocol_a": (
            "Stratified split of original images; on-the-fly augmentation "
            "applied to training only."
        ),
        "paper": {
            "correct": f"{PAPER_K}/{PAPER_N}",
            "accuracy": PAPER_ACC,
            "ci95": PAPER_CI,
        },
        "results": json_safe_records(table.drop(columns=["note"])),
        "ablation": ablation_rows,
        "protocol_b": protocol_b_summary,
        "protocol_b_prime": protocol_b_prime_summary,
        "duplicate_groups": group_summary,
        "duplicate_audits": duplicate_audits,
        "validation_selection": json_safe_records(validation_table),
        "ensemble_agreement": agreement,
    }
    save_json(cfg.out_path / "summary.json", summary)
    print(f"\nSaved results to {cfg.out_path}")


if __name__ == "__main__":
    main()
