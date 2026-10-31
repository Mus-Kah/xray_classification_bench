from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def prep(xb: torch.Tensor) -> torch.Tensor:
    """Normalize a uint8/float tensor in 0..255 with ImageNet statistics."""
    mean = torch.tensor(IMAGENET_MEAN, device=xb.device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=xb.device, dtype=torch.float32).view(1, 3, 1, 1)
    return (xb.float() / 255.0 - mean) / std


def gpu_aug(xb: torch.Tensor, rotation: float = 15.0) -> torch.Tensor:
    """On-GPU training augmentation.

    Applies a random-resized crop (area 0.5-1.0, ratio 0.9-1.1), horizontal flip
    with probability 0.5, rotation in +/-rotation degrees, and brightness/contrast
    jitter in [0.8, 1.2]. Input is uint8 (B, 3, H, W); output is float32 in [0, 255].
    """
    batch_size = xb.shape[0]
    device = xb.device
    x = xb.float()

    def uniform(low: float, high: float) -> torch.Tensor:
        return torch.empty(batch_size, device=device).uniform_(low, high)

    area = uniform(0.5, 1.0)
    ratio = torch.exp(uniform(math.log(0.9), math.log(1 / 0.9)))
    width = (area * ratio).sqrt().clamp(max=1.0)
    height = (area / ratio).sqrt().clamp(max=1.0)
    center_x = (torch.rand(batch_size, device=device) * 2 - 1) * (1 - width)
    center_y = (torch.rand(batch_size, device=device) * 2 - 1) * (1 - height)
    angle = uniform(-rotation, rotation) * math.pi / 180
    flip = 1 - 2 * (torch.rand(batch_size, device=device) < 0.5).float()
    cos, sin = angle.cos(), angle.sin()

    theta = torch.stack(
        [
            torch.stack([width * cos * flip, -width * sin, center_x], dim=1),
            torch.stack([height * sin * flip, height * cos, center_y], dim=1),
        ],
        dim=1,
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    x = x * uniform(0.8, 1.2).view(batch_size, 1, 1, 1)
    channel_mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - channel_mean) * uniform(0.8, 1.2).view(batch_size, 1, 1, 1) + channel_mean
    return x.clamp_(0, 255)


def offline_augment(
    image: np.ndarray, rng: np.random.RandomState, rotation: float = 30.0
) -> np.ndarray:
    """Deterministic offline augmentation used by the augment-before-split protocol."""
    size = image.shape[0]
    angle = rng.uniform(-rotation, rotation)
    flip = rng.rand() < 0.5
    crop = int(round(size * rng.uniform(0.5, 1.0)))
    x0 = rng.randint(0, size - crop + 1)
    y0 = rng.randint(0, size - crop + 1)
    out = cv2.resize(
        image[y0 : y0 + crop, x0 : x0 + crop], (size, size), interpolation=cv2.INTER_LINEAR
    )
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    out = cv2.warpAffine(out, matrix, (size, size), borderMode=cv2.BORDER_REFLECT)
    if flip:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)
