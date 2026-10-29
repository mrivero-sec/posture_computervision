"""Exporte le classifieur entraine en en-tete C pour l'ESP32-S3.

  python scripts/export_c.py   ->  firmware/main/posture_model.h
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

import _common  # noqa: F401
from _common import ROOT

from posture.features import FEATURE_NAMES, N_FEATURES


def arr(name: str, a: np.ndarray) -> str:
    flat = ", ".join(f"{float(v):.9g}f" for v in np.asarray(a, dtype=np.float64).reshape(-1))
    return f"static const float {name}[{a.size}] = {{{flat}}};\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "posture_clf.joblib"))
    ap.add_argument("--out", default=str(ROOT / "firmware" / "main" / "posture_model.h"))
    args = ap.parse_args()

    b = joblib.load(args.model)
    assert b["features"] == FEATURE_NAMES, "features differentes : reentrainer"
    pipe = b["pipeline"]
    sc, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]

    lines = [
        "// Fichier GENERE par scripts/export_c.py : ne pas editer a la main.\n",
        f"// Modele : {b['kind']}  |  entraine le {b['trained_at']}  |  {b['n_frames']} frames, "
        f"{b['n_sessions']} sessions  |  profil {b['profile']}\n",
        f"// Validation croisee par session : F1={b['cv'][b['kind']]['f1']:.3f}  "
        f"AUC={b['cv'][b['kind']]['roc_auc']:.3f}\n",
        "#pragma once\n\n",
        f"#define POSTURE_N_FEATURES {N_FEATURES}\n",
        f"#define POSTURE_MODEL_IS_MLP {1 if b['kind'] == 'mlp' else 0}\n",
        f"#define POSTURE_SUGGESTED_THRESHOLD {b['threshold']:.3f}f\n\n",
        "// Ordre des features : " + ", ".join(FEATURE_NAMES) + "\n",
        arr("POSTURE_MEAN", sc.mean_),
        arr("POSTURE_SCALE", sc.scale_),
    ]
    if b["kind"] == "logreg":
        lines += [arr("POSTURE_W", clf.coef_[0]), f"static const float POSTURE_B = {float(clf.intercept_[0]):.9g}f;\n"]
    else:
        w1, w2 = clf.coefs_
        h = w1.shape[1]
        lines += [
            f"#define POSTURE_HIDDEN {h}\n",
            "// W1 : [feature][neurone] (row-major)\n",
            arr("POSTURE_W1", w1), arr("POSTURE_B1", clf.intercepts_[0]),
            arr("POSTURE_W2", w2[:, 0]), f"static const float POSTURE_B2 = {float(clf.intercepts_[1][0]):.9g}f;\n",
        ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    print(f"En-tete ecrit : {out}")


if __name__ == "__main__":
    main()
