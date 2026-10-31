from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

CLASSES = ["Normal", "Lung Opacity", "Viral Pneumonia"]

DEFAULT_MODELS = [
    "convnext_tiny.fb_in22k_ft_in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
]


@dataclass
class Config:
    data_dir: str = "data"
    out_dir: str = "results"
    image_size: int = 224
    models: List[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    split_seed: int = 42
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-4
    head_mult: float = 10.0
    head_mult_by: Dict[str, float] = field(default_factory=dict)
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    ema_decay: float = 0.99
    aug_rotation: float = 15.0
    tta: bool = True
    quick_test: bool = False

    run_ablation: bool = True
    run_protocol_b: bool = True
    protocol_b_epochs: int = 8
    protocol_b_seeds: List[int] = field(default_factory=lambda: [0])

    def __post_init__(self) -> None:
        self.data_dir = str(self.data_dir)
        self.out_dir = str(self.out_dir)
        self.models = list(self.models)
        self.seeds = list(self.seeds)
        self.head_mult_by = dict(self.head_mult_by)
        self.protocol_b_seeds = list(self.protocol_b_seeds)
        if self.quick_test:
            self.seeds = [0]
            self.epochs = 2
            self.run_ablation = False
            self.run_protocol_b = False
            if not self.out_dir.endswith("_quicktest"):
                self.out_dir = self.out_dir + "_quicktest"

    @property
    def out_path(self) -> Path:
        return Path(self.out_dir)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: Optional[str | Path] = None, **overrides: Any) -> Config:
    data: Dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    data.update({key: value for key, value in overrides.items() if value is not None})
    return Config(**data)
