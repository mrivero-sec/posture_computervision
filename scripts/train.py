"""Trains and compares: ISO rules vs logistic regression vs MLP.

Cross-validation BY SESSION (GroupKFold): neighbouring frames are almost
identical, so splitting by frame would inflate the scores.

  python scripts/train.py                       # real data in data/
  python scripts/train.py --data data/synthetic # pipeline test
  python scripts/train.py --profile desk        # a single profile
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np

import _common  # noqa: F401
from _common import ROOT

from posture import features, rules
from posture.io import DATA_DIR, load_sessions


def build(kind: str, seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    elif kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(16,), activation="relu", alpha=1e-3,
                            max_iter=3000, early_stopping=True, random_state=seed)
    else:
        raise ValueError(kind)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def metrics(y, p, thr=0.5) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    yh = (p >= thr).astype(int)
    return {
        "f1": float(f1_score(y, yh, zero_division=0)),
        "precision_bad": float(precision_score(y, yh, zero_division=0)),
        "recall_bad": float(recall_score(y, yh, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
    }


def best_threshold(y, p) -> float:
    from sklearn.metrics import f1_score

    grid = np.linspace(0.2, 0.9, 71)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--profile", default=None)
    ap.add_argument("--out", default=str(ROOT / "models" / "posture_clf.joblib"))
    ap.add_argument("--force", choices=["logreg", "mlp"], default=None, help="force the model type")
    args = ap.parse_args()

    import joblib
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    df, kps = load_sessions(args.data)
    if args.profile:
        m = (df["profile"] == args.profile).to_numpy()
        df, kps = df[m].reset_index(drop=True), kps[m]
    present = np.array([features.person_present(k) for k in kps])
    df, kps = df[present].reset_index(drop=True), kps[present]
    X = features.compute_many(kps)          # recomputed: follows any change in features.py
    y = df["label"].to_numpy(dtype=int)
    groups = df["session"].to_numpy()
    n_sess = len(set(groups))
    print(f"{len(y)} frames, {n_sess} sessions, {int(y.sum())} bad / {int((1 - y).sum())} good")
    if len(set(y)) < 2:
        raise SystemExit("Both good AND bad examples are needed.")

    if n_sess >= 2:
        splits = list(GroupKFold(n_splits=min(5, n_sess)).split(X, y, groups))
    else:
        print("WARNING: only one session, validating by frame (optimistic scores).")
        splits = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))

    oof = {"rules": np.array([rules.score(f) for f in X]), "logreg": np.zeros(len(y)), "mlp": np.zeros(len(y))}
    for tr, te in splits:
        for kind in ("logreg", "mlp"):
            pipe = build(kind).fit(X[tr], y[tr])
            oof[kind][te] = pipe.predict_proba(X[te])[:, 1]

    report = {k: metrics(y, p) for k, p in oof.items()}
    print(f"\n{'model':8s} {'F1':>6s} {'prec':>6s} {'recall':>7s} {'AUC':>6s}")
    for k, r in report.items():
        print(f"{k:8s} {r['f1']:6.3f} {r['precision_bad']:6.3f} {r['recall_bad']:7.3f} {r['roc_auc']:6.3f}")

    if args.force:
        kind = args.force
    else:  # the simpler model wins if within 0.02 F1
        kind = "logreg" if report["logreg"]["f1"] >= report["mlp"]["f1"] - 0.02 else "mlp"
    thr = best_threshold(y, oof[kind])
    final = build(kind).fit(X, y)
    bundle = {
        "pipeline": final, "kind": kind, "features": features.FEATURE_NAMES, "threshold": thr,
        "cv": report, "n_frames": int(len(y)), "n_sessions": int(n_sess),
        "trained_at": datetime.now().isoformat(timespec="seconds"), "profile": args.profile or "all",
    }
    from pathlib import Path

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    meta = {k: v for k, v in bundle.items() if k != "pipeline"}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSelected model: {kind} (best F1 threshold in validation = {thr:.2f}) -> {out}")
    print("Next step: python scripts/export_c.py")


if __name__ == "__main__":
    main()
