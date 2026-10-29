"""Features posturales calculees depuis 17 keypoints COCO.

IMPORTANT : ce fichier est le miroir exact de firmware/main/features.cpp.
Toute modification doit etre reportee des deux cotes (le test
tests/test_parity.py compare les deux implementations).

Conventions : image en pixels, y vers le bas. Les angles sont en degres.
"Avant" = direction vers laquelle regarde la personne (nez devant l'oreille).
"""
from __future__ import annotations

import math

import numpy as np

from .keypoints import (L_EAR, L_EYE, L_HIP, L_SH, NOSE, R_EAR, R_EYE, R_HIP, R_SH)

FEATURE_NAMES = [
    "neck_incl",      # angle epaule->oreille vs verticale (+ = tete en avant)
    "trunk_incl",     # angle hanche->epaule vs verticale (+ = penche en avant)
    "neck_rel",       # neck_incl - trunk_incl (flexion cervicale)
    "head_fwd",       # decalage horizontal oreille/epaule / longueur de reference
    "nose_drop",      # nez sous l'oreille / longueur de reference (regard bas)
    "shoulder_tilt",  # |angle ligne des epaules vs horizontale|
    "head_tilt",      # |angle ligne des yeux (ou oreilles) vs horizontale|
    "lateral_lean",   # |angle milieu hanches -> milieu epaules vs verticale|
    "ear_sh_gap",     # hauteur oreille au-dessus de l'epaule / longueur de reference
    "nose_offset",    # |nez - milieu epaules| horizontal / longueur de reference
    "view_ratio",     # largeur d'epaules / longueur de reference (0 = profil)
    "hips_visible",   # 1 si au moins une hanche est detectee
]
N_FEATURES = len(FEATURE_NAMES)
CONF_THR = 0.5
EPS = 1e-6


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _deg(rad: float) -> float:
    return rad * 180.0 / math.pi


class _P:
    __slots__ = ("x", "y", "ok")

    def __init__(self, x=0.0, y=0.0, ok=False):
        self.x, self.y, self.ok = float(x), float(y), bool(ok)


def _pt(kps, i, thr) -> _P:
    x, y, c = float(kps[i][0]), float(kps[i][1]), float(kps[i][2])
    ok = c >= thr and not (x == 0.0 and y == 0.0)
    return _P(x, y, ok)


def _mid(a: _P, b: _P) -> _P:
    if a.ok and b.ok:
        return _P((a.x + b.x) / 2.0, (a.y + b.y) / 2.0, True)
    if a.ok:
        return _P(a.x, a.y, True)
    if b.ok:
        return _P(b.x, b.y, True)
    return _P()


def _dist(a: _P, b: _P) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def person_present(kps, thr: float = CONF_THR) -> bool:
    """Assez de points pour juger : (nez ou une oreille) + une epaule."""
    head = _pt(kps, NOSE, thr).ok or _pt(kps, L_EAR, thr).ok or _pt(kps, R_EAR, thr).ok
    sh = _pt(kps, L_SH, thr).ok or _pt(kps, R_SH, thr).ok
    return head and sh


def compute(kps, thr: float = CONF_THR) -> np.ndarray:
    """kps : (17, 3) [x, y, conf]. Retourne un vecteur float64 de N_FEATURES."""
    nose = _pt(kps, NOSE, thr)
    leye, reye = _pt(kps, L_EYE, thr), _pt(kps, R_EYE, thr)
    lear, rear = _pt(kps, L_EAR, thr), _pt(kps, R_EAR, thr)
    lsh, rsh = _pt(kps, L_SH, thr), _pt(kps, R_SH, thr)
    lhip, rhip = _pt(kps, L_HIP, thr), _pt(kps, R_HIP, thr)

    # cote le plus visible pour la vue de profil (egalite -> gauche)
    score_l = int(lear.ok) + int(lsh.ok) + int(lhip.ok)
    score_r = int(rear.ok) + int(rsh.ok) + int(rhip.ok)
    if score_l >= score_r:
        ear, sh, hip = lear, lsh, lhip
        ear2, sh2, hip2 = rear, rsh, rhip
    else:
        ear, sh, hip = rear, rsh, rhip
        ear2, sh2, hip2 = lear, lsh, lhip
    if not ear.ok:
        ear = ear2
    if not sh.ok:
        sh = sh2
    if not hip.ok:
        hip = hip2

    mid_sh = _mid(lsh, rsh)
    mid_hip = _mid(lhip, rhip)
    mid_ear = _mid(lear, rear)
    hips_visible = 1.0 if mid_hip.ok else 0.0

    # direction du regard (+1 = vers x croissant)
    dirx = 1.0
    if nose.ok and ear.ok and abs(nose.x - ear.x) > EPS:
        dirx = 1.0 if nose.x > ear.x else -1.0
    elif nose.ok and sh.ok and abs(nose.x - sh.x) > EPS:
        dirx = 1.0 if nose.x > sh.x else -1.0
    else:
        # nez cache (main devant la bouche...) : yeux, sinon on suppose la tete
        # en avant de l'epaule (sinon l'angle s'inverserait et paraitrait "bon")
        eye = _mid(leye, reye)
        if eye.ok and ear.ok and abs(eye.x - ear.x) > EPS:
            dirx = 1.0 if eye.x > ear.x else -1.0
        elif ear.ok and sh.ok and abs(ear.x - sh.x) > EPS:
            dirx = 1.0 if ear.x > sh.x else -1.0

    # longueur de reference (invariance a l'echelle)
    sh_w = _dist(lsh, rsh) if (lsh.ok and rsh.ok) else 0.0
    if mid_hip.ok and mid_sh.ok:
        L = _dist(mid_sh, mid_hip)
    elif ear.ok and sh.ok:
        L = 2.5 * _dist(ear, sh)
    elif sh_w > 0.0:
        L = sh_w
    else:
        L = 1.0
    if L < EPS:
        L = 1.0

    neck_incl = 0.0
    head_fwd = 0.0
    if ear.ok and sh.ok:
        vx, vy = ear.x - sh.x, ear.y - sh.y
        neck_incl = _deg(math.atan2(dirx * vx, -vy))
        head_fwd = dirx * vx / L

    trunk_incl = 0.0
    if sh.ok and hip.ok:
        vx, vy = sh.x - hip.x, sh.y - hip.y
        trunk_incl = _deg(math.atan2(dirx * vx, -vy))

    neck_rel = neck_incl - trunk_incl if hips_visible > 0.0 else neck_incl

    nose_drop = (nose.y - ear.y) / L if (nose.ok and ear.ok) else 0.0

    shoulder_tilt = 0.0
    if lsh.ok and rsh.ok:
        shoulder_tilt = abs(_deg(math.atan2(lsh.y - rsh.y, abs(lsh.x - rsh.x) + EPS)))

    head_tilt = 0.0
    if leye.ok and reye.ok:
        head_tilt = abs(_deg(math.atan2(leye.y - reye.y, abs(leye.x - reye.x) + EPS)))
    elif lear.ok and rear.ok:
        head_tilt = abs(_deg(math.atan2(lear.y - rear.y, abs(lear.x - rear.x) + EPS)))

    lateral_lean = 0.0
    if mid_sh.ok and mid_hip.ok:
        lateral_lean = abs(_deg(math.atan2(mid_sh.x - mid_hip.x, -(mid_sh.y - mid_hip.y))))

    ear_sh_gap = (mid_sh.y - mid_ear.y) / L if (mid_sh.ok and mid_ear.ok) else 0.0
    nose_offset = abs(nose.x - mid_sh.x) / L if (nose.ok and mid_sh.ok) else 0.0
    view_ratio = sh_w / L

    f = [
        _clip(neck_incl, -90.0, 120.0),
        _clip(trunk_incl, -90.0, 120.0),
        _clip(neck_rel, -120.0, 150.0),
        _clip(head_fwd, -2.0, 2.0),
        _clip(nose_drop, -2.0, 2.0),
        _clip(shoulder_tilt, 0.0, 90.0),
        _clip(head_tilt, 0.0, 90.0),
        _clip(lateral_lean, 0.0, 120.0),
        _clip(ear_sh_gap, -2.0, 2.0),
        _clip(nose_offset, 0.0, 3.0),
        _clip(view_ratio, 0.0, 3.0),
        hips_visible,
    ]
    return np.asarray(f, dtype=np.float64)


def compute_many(kps_array: np.ndarray, thr: float = CONF_THR) -> np.ndarray:
    """kps_array : (N, 17, 3) -> (N, N_FEATURES)."""
    return np.stack([compute(k, thr) for k in kps_array]) if len(kps_array) else np.zeros((0, N_FEATURES))
