from __future__ import annotations

import math
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import ModelEmaV2

from .augment import prep
from .config import Config
from .data import GPULoader
from .models import make_model, predict
from .utils import set_seed, short_name


def result_path(cfg: Config, tag: str, model_name: str, seed: int, epochs: int) -> Path:
    return Path(cfg.out_dir) / f"{tag}_{short_name(model_name)}_s{seed}_e{epochs}_v2.pkl"


def train_one(
    cfg: Config,
    model_name: str,
    seed: int,
    items: tuple[np.ndarray, np.ndarray, np.ndarray],
    images: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    epochs: int,
    augment: bool = True,
    tag: str = "",
    checkpoint: Optional[str | Path] = None,
) -> dict:
    path = result_path(cfg, tag, model_name, seed, epochs)
    if path.exists():
        with open(path, "rb") as handle:
            return pickle.load(handle)

    set_seed(seed)
    train_items, val_items, test_items = items
    train_loader = GPULoader(
        images,
        labels,
        train_items,
        cfg.batch_size,
        device,
        shuffle=True,
        augment=augment,
        aug_rotation=cfg.aug_rotation,
    )
    val_loader = GPULoader(images, labels, val_items, cfg.batch_size, device)
    test_loader = GPULoader(images, labels, test_items, cfg.batch_size, device)

    model = make_model(model_name).to(device)
    model.eval()
    with torch.no_grad():
        xb0, yb0 = next(iter(val_loader))
        initial_loss = F.cross_entropy(model(prep(xb0)).float(), yb0).item()
    print(f"[{tag} {short_name(model_name)} s{seed}] initial loss {initial_loss:.2f} (chance = 1.10)")

    ema = ModelEmaV2(model, decay=cfg.ema_decay)
    head_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
    groups = [
        {
            "params": [p for p in model.parameters() if id(p) not in head_ids],
            "lr": cfg.lr,
        },
        {
            "params": [p for p in model.parameters() if id(p) in head_ids],
            "lr": cfg.lr
            * cfg.head_mult_by.get(short_name(model_name), cfg.head_mult),
        },
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / steps_per_epoch
        if step < steps_per_epoch
        else 0.5
        * (
            1
            + math.cos(
                math.pi * (step - steps_per_epoch) / max(1, total_steps - steps_per_epoch)
            )
        ),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    y_val = labels[val_items[:, 0]]
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss, seen = 0.0, 0
        started = time.time()
        for xb, yb in train_loader:
            xb, yb = prep(xb.to(device, non_blocking=True)), yb.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                loss = criterion(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            train_loss += loss.item() * len(yb)
            seen += len(yb)

        val_probs = predict(ema.module, val_loader, tta=False)
        val_acc = float((val_probs.argmax(1) == y_val).mean())
        val_loss = float(-np.log(val_probs[np.arange(len(y_val)), y_val] + 1e-9).mean())
        history.append((train_loss / seen, val_loss, val_acc))
        print(
            f"[{tag} {short_name(model_name)} s{seed}] ep{epoch + 1:02d} "
            f"train_loss {train_loss / seen:.3f} | val_loss {val_loss:.3f} "
            f"val_acc {val_acc:.3f} | {time.time() - started:.0f}s"
        )
        if epoch == 0 and train_loss / seen > 2.0:
            print(
                "   WARNING: epoch-1 train loss > 2 (3-class chance is ~1.1): "
                "training looks unstable for this model."
            )

    result = {
        "probs_te": predict(ema.module, test_loader, cfg.tta),
        "probs_va": predict(ema.module, val_loader, cfg.tta),
        "hist": history,
        "n_params": sum(parameter.numel() for parameter in model.parameters()) / 1e6,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(result, handle)

    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ema.module.state_dict(), checkpoint)

    del model, ema, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
