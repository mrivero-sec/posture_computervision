"""Lecture / ecriture des sessions de collecte (CSV, sans images)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES
from .keypoints import COCO_NAMES

KP_COLS = [f"{n}_{c}" for n in COCO_NAMES for c in ("x", "y", "c")]
META_COLS = ["session", "t", "profile", "label"]
ALL_COLS = META_COLS + KP_COLS + FEATURE_NAMES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class SessionWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        self.fh = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.fh)
        if new:
            self.w.writerow(ALL_COLS)

    def write(self, session: str, t: float, profile: str, label: int, kps: np.ndarray, feats: np.ndarray):
        row = [session, f"{t:.3f}", profile, int(label)]
        row += [f"{v:.2f}" if i % 3 != 2 else f"{v:.3f}" for i, v in enumerate(kps.reshape(-1))]
        row += [f"{v:.5f}" for v in feats]
        self.w.writerow(row)

    def close(self):
        self.fh.close()


def load_sessions(data_dir: Path | str = DATA_DIR, pattern: str = "*.csv"):
    """Retourne (df, kps) ; kps de forme (N, 17, 3)."""
    import pandas as pd

    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucune session dans {data_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["session"] = df["session"].astype(str)
    kps = df[KP_COLS].to_numpy(dtype=np.float64).reshape(-1, 17, 3)
    return df, kps
