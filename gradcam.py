from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch

from cxrbench.augment import prep
from cxrbench.config import CLASSES, load_config
from cxrbench.data import find_images, load_images, split_clean
from cxrbench.explain import gradcam
from cxrbench.utils import describe_device, resolve_device, short_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM examples for a trained ConvNeXt model."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--cpu", action="store_true", help="Allow CPU execution (debugging only)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(
        args.config, data_dir=args.data_dir, out_dir=args.out_dir
    )
    device = resolve_device(require_cuda=not args.cpu)
    print(f"Device: {device} | {describe_device(device)}")

    model_name = args.model or cfg.models[0]
    if not short_name(model_name).startswith("convnext"):
        raise SystemExit(
            "Grad-CAM is configured for ConvNeXt models because their stage "
            "output uses the (B, C, H, W) layout expected by the implementation."
        )

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else Path(cfg.out_dir) / f"ckpt_{short_name(model_name)}_s{args.seed}.pt"
    )
    if not checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint}\n"
            "Run train.py first; seed 0 saves a checkpoint for each model."
        )

    df = find_images(cfg.data_dir)
    if df.empty:
        raise SystemExit(
            f"No images found under {cfg.data_dir!r}. See data/README.md for the expected layout."
        )
    labels = df["y"].to_numpy()
    _, clahe_images = load_images(
        df, cfg.image_size, Path(cfg.out_dir) / f"X_{cfg.image_size}.npz"
    )
    items_a = split_clean(labels, cfg.split_seed)
    test_ids = items_a[2][:, 0]

    model = timm.create_model(
        short_name(model_name), pretrained=False, num_classes=len(CLASSES)
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    layer = model.stages[-1]
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for class_index in range(len(CLASSES)):
        class_ids = test_ids[labels[test_ids] == class_index]
        for sample_index in range(2):
            image_index = int(rng.choice(class_ids))
            tensor = (
                prep(
                    torch.from_numpy(clahe_images[image_index])
                    .permute(2, 0, 1)[None]
                    .to(device)
                )
            )
            cam, prediction = gradcam(model, tensor, layer)
            image_axis = axes[class_index, 2 * sample_index]
            cam_axis = axes[class_index, 2 * sample_index + 1]
            image_axis.imshow(clahe_images[image_index])
            image_axis.set_title(
                f"true: {CLASSES[class_index]}", fontsize=8
            )
            image_axis.axis("off")
            cam_axis.imshow(clahe_images[image_index])
            cam_axis.imshow(cam[0], cmap="jet", alpha=0.45)
            cam_axis.set_title(
                f"Grad-CAM, pred: {CLASSES[int(prediction[0])]}", fontsize=8
            )
            cam_axis.axis("off")

    fig.tight_layout()
    output_path = Path(cfg.out_dir) / "gradcam.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM figure to {output_path}")


if __name__ == "__main__":
    main()
