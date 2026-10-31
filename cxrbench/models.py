from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn as nn

from .augment import prep


def make_model(
    name: str,
    num_classes: int = 3,
    drop_rate: float = 0.2,
    drop_path_rate: float = 0.1,
) -> nn.Module:
    try:
        model = timm.create_model(
            name,
            pretrained=True,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
    except Exception as exc:
        print(f"Pretrained tag failed ({exc}); falling back to the base tag.")
        model = timm.create_model(
            name.split(".")[0],
            pretrained=True,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    if "efficientnet" in name:
        head = model.get_classifier()
        nn.init.normal_(head.weight, std=0.01)
        nn.init.zeros_(head.bias)
    return model


@torch.no_grad()
def predict(model: nn.Module, loader, tta: bool = True) -> np.ndarray:
    model.eval()
    probabilities = []
    device = next(model.parameters()).device
    for xb, _ in loader:
        xb = prep(xb.to(device, non_blocking=True))
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            probs = model(xb).float().softmax(1)
            if tta:
                probs = (probs + model(torch.flip(xb, [3])).float().softmax(1)) / 2
        probabilities.append(probs.cpu())
    return torch.cat(probabilities).numpy()
