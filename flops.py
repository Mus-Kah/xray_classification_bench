from __future__ import annotations

import argparse

import pandas as pd

from cxrbench.config import load_config
from cxrbench.models import count_macs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report multiply-accumulate operations for the configured backbones."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(
        args.config, out_dir=args.out_dir, models=args.models
    )

    rows = []
    for name in cfg.models:
        macs = count_macs(name, image_size=cfg.image_size)
        rows.append({"model": name, "macs_g": macs})
        print(f"{name:45s} {macs:.2f} G (MACs)")

    output_path = cfg.out_path / "flops.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved MACs to {output_path}")


if __name__ == "__main__":
    main()
