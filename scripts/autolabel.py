"""Pre-labellise des sessions avec les regles ISO (colonne `label_rules`)
et liste les frames ou ton label et les regles divergent, pour verification.

  python scripts/autolabel.py data/bureau_01.csv
"""
from __future__ import annotations

import argparse

import numpy as np

import _common  # noqa: F401

from posture import config, features, rules
from posture.io import KP_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--write", action="store_true", help="ajoute label_rules et score_rules au CSV")
    args = ap.parse_args()
    import pandas as pd

    cfg = config.load()
    for path in args.csv:
        df = pd.read_csv(path)
        kps = df[KP_COLS].to_numpy(dtype=np.float64).reshape(-1, 17, 3)
        F = features.compute_many(kps)
        s = np.array([rules.score(f, cfg["rules"]) for f in F])
        df["score_rules"] = s.round(3)
        df["label_rules"] = (s >= 0.5).astype(int)
        df["worst_rule"] = [rules.worst(f, cfg["rules"]) for f in F]
        agree = (df["label_rules"] == df["label"]).mean()
        print(f"{path}: accord regles / labels manuels = {agree:.1%}")
        dis = df[df["label_rules"] != df["label"]]
        if len(dis):
            print(dis[["t", "label", "label_rules", "score_rules", "worst_rule"]].head(20).to_string(index=False))
        if args.write:
            df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
