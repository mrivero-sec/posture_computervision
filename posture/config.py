from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["rules"] = {k: tuple(v) for k, v in cfg.get("rules", {}).items()}
    return cfg
