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

from cxrbench.config import CLASSES, load_config
from cxrbench.data import (
    duplicate_report,
    find_images,
    load_images,
    sibling_overlap,
    source_overlap,
    split_augment_then_split,
    split_clean,
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

    duplicates = duplicate_report(df)
    print(f"Exact duplicate files: {len(duplicates)} | groups: {duplicates['h'].nunique()}")
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
    }
    save_json(cfg.out_path / "summary.json", summary)
    print(f"\nSaved results to {cfg.out_path}")


if __name__ == "__main__":
    main()
