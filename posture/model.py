"""Chargement / prediction du classifieur (ou repli sur les regles)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import rules
from .features import FEATURE_NAMES

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "posture_clf.joblib"


class PostureModel:
    def __init__(self, path: str | Path | None = None, rule_thresholds: dict | None = None):
        self.rule_thresholds = rule_thresholds
        self.path = Path(path) if path else DEFAULT_MODEL
        self.clf = None
        self.kind = "rules"
        if self.path.exists():
            import joblib

            bundle = joblib.load(self.path)
            if bundle.get("features") != FEATURE_NAMES:
                raise ValueError("Le modele a ete entraine avec d'autres features : relancer scripts/train.py")
            self.clf = bundle["pipeline"]
            self.kind = bundle.get("kind", "sklearn")
            self.threshold = float(bundle.get("threshold", 0.5))

    def predict_proba(self, f: np.ndarray) -> float:
        if self.clf is None:
            return rules.score(f, self.rule_thresholds)
        return float(self.clf.predict_proba(np.asarray(f, dtype=np.float64).reshape(1, -1))[0, 1])
