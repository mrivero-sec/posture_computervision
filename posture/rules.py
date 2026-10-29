"""Baseline par regles (zero entrainement), inspiree d'ISO 11226 / EN 1005-4.

- tete inclinee vers l'avant > 25 deg  -> hors zone faible risque
- flexion cervicale (neck_rel) > 25 deg -> hors zone faible risque
- tronc incline > 20 deg               -> hors zone faible risque
- inclinaison laterale de la tete > 10 deg

Approximation : l'axe epaule->oreille sert de proxy a l'inclinaison de la tete.
Le score renvoye vaut 0.5 pile au seuil et monte quand on le depasse.
"""
from __future__ import annotations

import math

from .features import FEATURE_NAMES

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}

DEFAULT_THRESHOLDS = {
    "neck_incl": (25.0, 10.0),   # (seuil, echelle en deg pour la pente du score)
    "neck_rel": (25.0, 10.0),
    "trunk_incl": (20.0, 10.0),
    "head_tilt": (10.0, 5.0),
    "shoulder_tilt": (10.0, 5.0),
}


def exceedances(f, thresholds=None) -> dict:
    t = thresholds or DEFAULT_THRESHOLDS
    out = {}
    side_view = f[IDX["view_ratio"]] < 0.5  # epaules superposees -> profil
    for name, (thr, scale) in t.items():
        if name in ("head_tilt", "shoulder_tilt") and side_view:
            continue  # peu fiable de profil
        if name == "trunk_incl" and f[IDX["hips_visible"]] < 0.5:
            continue  # pas de hanches -> pas de tronc
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
