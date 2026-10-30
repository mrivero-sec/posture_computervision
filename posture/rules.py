"""Rule-based baseline (no training), inspired by ISO 11226 / EN 1005-4.

- head tilted forward > 25 deg        -> outside the low-risk zone
- neck flexion (neck_rel) > 25 deg    -> outside the low-risk zone
- trunk inclined > 20 deg             -> outside the low-risk zone
- sideways head tilt > 10 deg

Approximation: the shoulder->ear axis is used as a proxy for head inclination.
The returned score is exactly 0.5 at the threshold and rises above it.
"""
from __future__ import annotations

import math

from .features import FEATURE_NAMES

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}

DEFAULT_THRESHOLDS = {
    "neck_incl": (25.0, 10.0),   # (threshold, scale in deg for the score slope)
    "neck_rel": (25.0, 10.0),
    "trunk_incl": (20.0, 10.0),
    "head_tilt": (10.0, 5.0),
    "shoulder_tilt": (10.0, 5.0),
}


def exceedances(f, thresholds=None) -> dict:
    t = thresholds or DEFAULT_THRESHOLDS
    out = {}
    side_view = f[IDX["view_ratio"]] < 0.5  # shoulders overlap -> side view
    for name, (thr, scale) in t.items():
        if name in ("head_tilt", "shoulder_tilt") and side_view:
            continue  # unreliable in side view
        if name == "trunk_incl" and f[IDX["hips_visible"]] < 0.5:
            continue  # no hips -> no trunk
        out[name] = (f[IDX[name]] - thr) / scale
    return out


def score(f, thresholds=None) -> float:
    ex = exceedances(f, thresholds)
    if not ex:
        return 0.0
    s = max(ex.values())
    return 1.0 / (1.0 + math.exp(-2.0 * s))


def worst(f, thresholds=None) -> str:
    ex = exceedances(f, thresholds)
    return max(ex, key=ex.get) if ex else ""
