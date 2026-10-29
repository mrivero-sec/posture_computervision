"""Genere des sessions SYNTHETIQUES (squelettes 2D parametriques) pour tester
le pipeline collecte -> entrainement -> export C sans webcam.

Ces donnees ne servent PAS a evaluer la vraie performance : elles valident
que tout fonctionne de bout en bout. Les vraies donnees viennent de collect.py.

  python scripts/make_synthetic.py --sessions 6 --frames 600
"""
from __future__ import annotations

import argparse
import math

import numpy as np

import _common  # noqa: F401

from posture import features
from posture.io import DATA_DIR, SessionWriter
from posture.keypoints import (L_EAR, L_EYE, L_HIP, L_SH, NOSE, R_EAR, R_EYE, R_HIP, R_SH, empty_kps)


def _rot(dx, dy, deg):
    a = math.radians(deg)
    return dx * math.cos(a) - dy * math.sin(a), dx * math.sin(a) + dy * math.cos(a)


def skeleton(rng, view, trunk, neck_rel, pitch, head_tilt, sh_tilt, hips_hidden):
    k = empty_kps()
    s = rng.uniform(0.7, 1.4)                       # echelle (distance camera)
    T, N, W = 180 * s, 70 * s, 150 * s              # tronc, cou+tete, largeur epaules
    hx, hy = rng.uniform(250, 400), rng.uniform(380, 440)
    c = lambda: rng.uniform(0.75, 0.99)             # confiance keypoint

    if view == "side":
        d = rng.choice([-1.0, 1.0])                 # regarde a gauche ou a droite
        shx, shy = hx + d * T * math.sin(math.radians(trunk)), hy - T * math.cos(math.radians(trunk))
        na = trunk + neck_rel
        ex, ey = shx + d * N * math.sin(math.radians(na)), shy - N * math.cos(math.radians(na))
        nx, ny = ex + d * 0.45 * N * math.cos(math.radians(pitch)), ey + 0.45 * N * math.sin(math.radians(pitch))
        near = rng.choice([0, 1])                   # cote vers la camera
        o = 4 * s
        for i_ear, i_sh, i_hip, side in ((L_EAR, L_SH, L_HIP, 0), (R_EAR, R_SH, R_HIP, 1)):
            vis = side == near
            k[i_ear] = (ex + o * side, ey, c() if vis else rng.uniform(0.0, 0.6))
            k[i_sh] = (shx + o * side, shy, c() if vis else rng.uniform(0.3, 0.8))
            k[i_hip] = (hx + o * side, hy, c() if vis else rng.uniform(0.3, 0.8))
        k[NOSE] = (nx, ny, c())
        k[L_EYE] = (nx - d * 0.15 * N, ny - 0.15 * N, c())
        k[R_EYE] = (nx - d * 0.15 * N, ny - 0.15 * N, rng.uniform(0.0, 0.6))
    else:  # front
        lean = rng.normal(0, 3) + (trunk - 10) * 0.3 * rng.choice([-1, 1])
        mx, my = hx + T * math.sin(math.radians(lean)), hy - T * math.cos(math.radians(lean))
        dx, dy = _rot(W / 2, 0, sh_tilt * rng.choice([-1, 1]))
        k[L_SH] = (mx + dx, my + dy, c())
        k[R_SH] = (mx - dx, my - dy, c())
        drop = 0.25 * N * math.sin(math.radians(max(neck_rel, 0)))  # tete en avant = tete plus basse
        hcx, hcy = mx, my - N + drop
        ht = head_tilt * rng.choice([-1, 1])
        for idx, (ox, oy) in ((L_EYE, (0.18 * N, 0)), (R_EYE, (-0.18 * N, 0)),
                              (L_EAR, (0.38 * N, 0.08 * N)), (R_EAR, (-0.38 * N, 0.08 * N)),
                              (NOSE, (0, 0.18 * N))):
            rx, ry = _rot(ox, oy, ht)
            k[idx] = (hcx + rx, hcy + ry, c())
        k[L_HIP] = (hx + 0.35 * W, hy, c())
        k[R_HIP] = (hx - 0.35 * W, hy, c())
    if hips_hidden:
        k[L_HIP] = (0, 0, 0)
        k[R_HIP] = (0, 0, 0)
    k[:, :2] += rng.normal(0, 2.0, size=(17, 2)) * (k[:, 2:3] > 0)
    return k


def sample(rng, profile):
    bad = rng.random() < 0.5
    view = "side" if rng.random() < (0.7 if profile == "desk" else 0.6) else "front"
    if bad:
        kind = rng.choice(["neck", "trunk", "lat"] if view == "front" else ["neck", "trunk", "both"])
        trunk = rng.uniform(25, 45) if kind in ("trunk", "both") else rng.uniform(-5, 15)
        neck_rel = rng.uniform(30, 55) if kind in ("neck", "both") else rng.uniform(5, 20)
        head_tilt = rng.uniform(12, 25) if kind == "lat" else rng.uniform(0, 6)
        sh_tilt = rng.uniform(10, 18) if kind == "lat" else rng.uniform(0, 5)
    else:
        trunk, neck_rel = rng.uniform(-8, 14), rng.uniform(0, 20)
        head_tilt, sh_tilt = rng.uniform(0, 7), rng.uniform(0, 6)
    if profile == "practitioner" and not bad:
        trunk += rng.uniform(0, 5)
    pitch = rng.uniform(5, 25) + (15 if bad else 0)
    hips_hidden = rng.random() < (0.3 if profile == "desk" else 0.1)
    label = int(bad) if rng.random() > 0.03 else int(not bad)   # 3 % de bruit de label
    return skeleton(rng, view, trunk, neck_rel, pitch, head_tilt, sh_tilt, hips_hidden), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA_DIR / "synthetic"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    from pathlib import Path

    out = Path(args.out)
    for s in range(args.sessions):
        profile = "desk" if s % 2 == 0 else "practitioner"
        name = f"synth_{profile}_{s:02d}"
        w = SessionWriter(out / f"{name}.csv")
        for i in range(args.frames):
            kps, label = sample(rng, profile)
            w.write(name, i * 0.2, profile, label, kps, features.compute(kps))
        w.close()
    print(f"{args.sessions} sessions synthetiques ecrites dans {out}")


if __name__ == "__main__":
    main()
