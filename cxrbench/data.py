from __future__ import annotations

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .augment import gpu_aug, offline_augment

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_images(root: str | Path) -> pd.DataFrame:
    rows = []
    for path in Path(root).rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parent = path.parent.name.lower()
        if "normal" in parent:
            label = 0
        elif "opac" in parent:
            label = 1
        elif "viral" in parent or "pneu" in parent:
            label = 2
        else:
            continue
        rows.append((str(path), label))
    return (
        pd.DataFrame(rows, columns=["path", "y"])
        .sort_values("path")
        .reset_index(drop=True)
    )


def hash_images(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with an MD5 content hash column ``h``."""
    hashes = [hashlib.md5(Path(path).read_bytes()).hexdigest() for path in df["path"]]
    out = df.copy()
    out["h"] = hashes
    return out


def duplicate_report(df: pd.DataFrame) -> pd.DataFrame:
    hashed = hash_images(df)
    return hashed[hashed.duplicated("h", keep=False)]


def duplicate_group_summary(hashed: pd.DataFrame) -> dict:
    """Summarize exact-duplicate groups in a hash-annotated frame."""
    groups = [group for _, group in hashed.groupby("h") if len(group) > 1]
    return {
        "groups": len(groups),
        "files": int(sum(len(group) for group in groups)),
        "conflicting_label_groups": int(
            sum(group["y"].nunique() > 1 for group in groups)
        ),
    }


def duplicate_cross_split_report(hashed: pd.DataFrame, split_seed: int) -> dict:
    """Count exact-duplicate groups that span subsets of a clean split.

    ``hashed`` must contain the ``h`` column produced by :func:`hash_images`.
    """
    labels = hashed["y"].to_numpy()
    train, val, test = split_clean(labels, split_seed)
    part = {int(i): "train" for i in train[:, 0]}
    part.update({int(i): "val" for i in val[:, 0]})
    part.update({int(i): "test" for i in test[:, 0]})
    groups = [group for _, group in hashed.groupby("h") if len(group) > 1]
    spanning = [
        group for group in groups if len({part[int(i)] for i in group.index}) > 1
    ]
    train_test = [
        group
        for group in groups
        if {"train", "test"} <= {part[int(i)] for i in group.index}
    ]
    test_with_train_duplicate = int(
        sum(
            sum(part[int(i)] == "test" for i in group.index)
            for group in train_test
        )
    )
    return {
        "split_seed": split_seed,
        "groups_spanning_subsets": len(spanning),
        "train_test_groups": len(train_test),
        "test_images_with_train_duplicate": test_with_train_duplicate,
    }


def load_images(
    df: pd.DataFrame, size: int, cache_path: str | Path, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Load and resize images, returning the raw RGB and CLAHE versions."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = np.load(cache_path)
        if verbose:
            print(f"Loaded image cache: {cache_path}")
        return cached["raw"], cached["clahe"]

    raw_images, clahe_images = [], []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    iterator = tqdm(df["path"], desc="Preparing images", disable=not verbose)
    for path in iterator:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        rgb = cv2.cvtColor(
            cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        gray = clahe.apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        raw_images.append(rgb)
        clahe_images.append(np.stack([gray] * 3, axis=-1))

    raw = np.stack(raw_images)
    clahe_arr = np.stack(clahe_images)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, raw=raw, clahe=clahe_arr)
    return raw, clahe_arr


def _stack_items(ids: np.ndarray) -> np.ndarray:
    return np.stack([ids, np.zeros_like(ids)], axis=1)


def split_clean(
    labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split original images into train/val/test without source-image overlap."""
    indices = np.arange(len(labels))
    n_test = 1043 if len(labels) == 3475 else int(0.30 * len(labels))
    n_val = 244 if len(labels) == 3475 else int(0.07 * len(labels))
    train_val, test = train_test_split(
        indices, test_size=n_test, stratify=labels, random_state=seed
    )
    train, val = train_test_split(
        train_val, test_size=n_val, stratify=labels[train_val], random_state=seed
    )
    return _stack_items(train), _stack_items(val), _stack_items(test)


def split_augment_then_split(
    labels: np.ndarray, seed: int, n_variants: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference protocol: create offline variants first, then split the items."""
    items = np.array([(i, v) for i in range(len(labels)) for v in range(n_variants)])
    item_labels = labels[items[:, 0]]
    indices = np.arange(len(items))
    n_test = 4170 if len(labels) == 3475 else int(0.30 * len(items))
    n_val = 973 if len(labels) == 3475 else int(0.07 * len(items))
    train_val, test = train_test_split(
        indices, test_size=n_test, stratify=item_labels, random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=n_val,
        stratify=item_labels[train_val],
        random_state=seed,
    )
    return items[train], items[val], items[test]


def split_then_augment(
    labels: np.ndarray, seed: int, n_variants: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Protocol B' control: split source images first, then expand offline variants.

    The training budget and test-item composition match
    :func:`split_augment_then_split`, but no source image crosses a subset
    boundary, so any B - B' difference isolates sibling overlap.
    """
    train, val, test = split_clean(labels, seed)

    def expand(items: np.ndarray) -> np.ndarray:
        return np.array(
            [(int(i), variant) for i, _ in items for variant in range(n_variants)]
        )

    return expand(train), expand(val), expand(test)


def sibling_overlap(train_items: np.ndarray, test_items: np.ndarray) -> float:
    """Share of test items whose source image also appears in training."""
    train_ids = set(train_items[:, 0])
    return float(np.mean([int(i) in train_ids for i in test_items[:, 0]]))


def source_overlap(items_a: np.ndarray, items_b: np.ndarray) -> int:
    return len(set(items_a[:, 0]) & set(items_b[:, 0]))


class GPULoader:
    """In-memory batch loader with optional on-GPU augmentation."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        items: np.ndarray,
        batch_size: int,
        device: torch.device,
        shuffle: bool = False,
        augment: bool = False,
        aug_rotation: float = 15.0,
    ) -> None:
        self.images = images
        self.labels = labels
        self.items = items
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.augment = augment
        self.aug_rotation = aug_rotation

    def __len__(self) -> int:
        n_items = len(self.items)
        return n_items // self.batch_size if self.shuffle else math.ceil(n_items / self.batch_size)

    def __iter__(self):
        n_items = len(self.items)
        order = np.random.permutation(n_items) if self.shuffle else np.arange(n_items)
        for batch_index in range(len(self)):
            batch = self.items[
                order[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]
            ]
            image_ids = batch[:, 0]
            if (batch[:, 1] == 0).all():
                images = self.images[image_ids]
            else:
                images = np.stack(
                    [
                        self.images[i]
                        if variant == 0
                        else offline_augment(
                            self.images[i],
                            np.random.RandomState(int(i) * 17 + int(variant)),
                        )
                        for i, variant in batch
                    ]
                )
            xb = torch.from_numpy(images).to(self.device, non_blocking=True).permute(0, 3, 1, 2)
            if self.augment:
                xb = gpu_aug(xb, self.aug_rotation)
            yield xb, torch.from_numpy(self.labels[image_ids]).to(self.device)
