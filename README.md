# CXR Bench

A reproducible benchmark for three-class chest X-ray classification (Normal, Lung Opacity, Viral Pneumonia) using modern ImageNet-pretrained backbones.

The primary evaluation protocol splits the original images before any augmentation and applies augmentation on the fly to the training set only. The repository also includes a separately reported reference protocol that augments before splitting, so the effect of that design choice can be measured rather than assumed. Results are summarized with bootstrap confidence intervals and an exact binomial comparison against the published ATL-ResNeXt50 baseline.

## Overview

The project provides:

- A leakage-aware protocol (Protocol A) with a stratified split of original images.
- A disclosed augment-before-split reference protocol (Protocol B) for comparison.
- Five-seed training for ConvNeXt-T, EfficientNetV2-S, and Swin-T, plus seed and model ensembles.
- Bootstrap 95% confidence intervals, macro F1, confusion matrices, and per-class recall.
- Ablation of CLAHE and training-time augmentation.
- Grad-CAM examples for the ConvNeXt-T model.
- Cached result files so interrupted runs can resume.

## Dataset

The reference dataset contains 3,475 chest X-ray images:

| Class | Images |
| --- | ---: |
| Normal | 1,250 |
| Lung Opacity | 1,125 |
| Viral Pneumonia | 1,100 |

Download the dataset from Mendeley Data as described in the original publication and arrange it as follows:

```
data/
├── Normal/
├── Lung Opacity/
└── Viral Pneumonia/
```

Image files are not included in this repository. Folder matching is case-insensitive and accepts names containing `normal`, `opac`, and `viral` or `pneu`. See [`data/README.md`](data/README.md) for details.

## Installation

```bash
git clone https://github.com/Mus-Kah/xray_classification_bench
cd xray_classification_bench

python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip

# Install the CUDA build that matches your driver. Example for CUDA 12.4:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

A CUDA-capable GPU is required for realistic training runs. The reference run used an NVIDIA RTX 4050 Laptop GPU with 6 GB of memory. The `--cpu` flag is available for lightweight debugging only.

## Quick start

Run a two-epoch smoke test with one seed and no ablation or Protocol B:

```bash
python train.py --quick-test
```

Quick-test outputs are written to `results_quicktest/` so they cannot collide with a full run.

Run the full experiment:

```bash
python train.py --config configs/default.yaml
```

Evaluate cached results after training:

```bash
python evaluate.py --config configs/default.yaml
```

Generate Grad-CAM examples (seed 0 of the first model):

```bash
python gradcam.py --config configs/default.yaml
```

Useful overrides:

```bash
python train.py \
  --data-dir /path/to/data \
  --out-dir runs/full \
  --models convnext_tiny.fb_in22k_ft_in1k \
  --seeds 0 1 2 \
  --epochs 20
```

Use `--no-ablation` or `--no-protocol-b` to skip the optional experiments, and `--cpu` only for debugging.

## Protocols

### Protocol A: leakage-aware benchmark (primary)

- Use the original 3,475 images.
- Create a stratified train/validation/test split with sizes 2,188 / 244 / 1,043, matching the published split sizes.
- Apply augmentation on the fly to the training set only.
- Test images are never used for checkpoint selection; the final EMA weights are evaluated.

This protocol guarantees that no source image appears in both training and test sets.

### Protocol B: augment-before-split reference (secondary)

- Create four offline augmented variants of every image (13,900 items).
- Split the augmented items into train/validation/test.
- Train without additional on-the-fly augmentation.

This mirrors the augment-before-split design used in some published studies. Because variants of the same source image can land in different splits, Protocol B results are not directly comparable to Protocol A. The code reports the share of test items that have a sibling from the same source image in the training set, so the difference is visible and documented.

### Comparison with the published baseline

Protocol A uses the same dataset, task, and split sizes as the published ATL-ResNeXt50 study. The original split and predictions are not public, so the comparison is indicative rather than a controlled head-to-head test on identical images. The one-sided exact binomial test treats the published accuracy (978/1,043) as a fixed reference.

## Reference results

The following numbers come from one complete run of the accompanying notebook (five seeds per model, 20 epochs, 224 px inputs). They are provided as a reproducibility reference, not as a claim about identical test images.

| Model | Accuracy | 95% CI | Correct | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| ConvNeXt-T (5-seed ensemble) | 95.59% | 94.34–96.84 | 997/1,043 | 0.957 |
| EfficientNetV2-S (5-seed ensemble) | 93.58% | 92.04–95.01 | 976/1,043 | 0.937 |
| Swin-T (5-seed ensemble) | 95.01% | 93.67–96.26 | 991/1,043 | 0.951 |
| All-model ensemble | 95.59% | 94.34–96.84 | 997/1,043 | 0.957 |
| Published ATL-ResNeXt50 (non-augmented) | 93.77% | 92.13–95.16 | 978/1,043 | — |

Ablation for ConvNeXt-T (first three seeds):

| Variant | Mean accuracy | Std |
| --- | ---: | ---: |
| Full (CLAHE + on-the-fly augmentation) | 95.17% | 0.15 |
| No CLAHE | 95.24% | 0.29 |
| No augmentation | 94.31% | 0.56 |

Protocol comparison for ConvNeXt-T:

- Protocol A: 95.23%
- Protocol B: 96.35%
- Descriptive difference: +1.13 percentage points
- Share of Protocol B test items with a training sibling: 95.4%

The Protocol B difference is descriptive; the two protocols use different splits, so it should not be read as a causal estimate.

## Project structure

```
.
├── cxrbench/
│   ├── __init__.py
│   ├── augment.py        # GPU augmentation, normalization, offline augmentation
│   ├── config.py         # Dataclass configuration and YAML loader
│   ├── data.py           # Discovery, caching, splits, and the in-memory loader
│   ├── evaluation.py     # Result rows, bootstrap CIs, and classification summaries
│   ├── explain.py        # Grad-CAM
│   ├── models.py         # timm model construction and inference
│   ├── protocols.py      # Published reference numbers
│   ├── training.py       # Single training run with EMA, AMP, and cosine schedule
│   ├── utils.py          # Seeding, device resolution, JSON helpers
│   └── visualize.py      # Sample and confusion-matrix figures
├── configs/
│   └── default.yaml
├── data/
│   └── README.md
├── results/
│   └── .gitkeep
├── tests/
│   ├── test_data.py
│   └── test_protocols.py
├── train.py
├── evaluate.py
├── gradcam.py
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
└── .gitignore
```

## Reproducibility notes

- The split seed is fixed at 42; only the training seeds vary across runs.
- Training uses AdamW with a separate learning rate for the classifier head, label smoothing, EMA weights, mixed precision, gradient clipping, and a cosine schedule with warmup.
- The test set is never used for model selection.
- Exact duplicate image files are detected and reported.
- Result files are cached by model, seed, and epoch count. Delete the corresponding `.pkl` files to rerun a configuration.
- `data/`, `results/`, model checkpoints, and cached tensors are excluded from version control by default.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

Please also follow the license and attribution requirements of the original dataset and the referenced publication.
