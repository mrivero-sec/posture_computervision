"""Collects data from the ESP32-S3 board over USB serial (real product camera and angle).

The firmware sends one `KP,<ms>,x0,y0,...,x16,y16` line per frame. This script
labels them from the keyboard in a small window and writes them in the same
format as collect.py, so train.py can use them directly.

  pip install pyserial
  python scripts/serial_collect.py --port COM5 --profile practitioner

Keys (window focused): g = good, b = bad, space = pause, q = quit.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import numpy as np

import _common  # noqa: F401
from _common import put

from posture import features, rules
from posture.io import DATA_DIR, SessionWriter


def parse_kp(line: str):
    parts = line.strip().split(",")
    if len(parts) != 2 + 34 or parts[0] != "KP":
        return None
    xy = np.array([float(v) for v in parts[2:]], dtype=np.float32).reshape(17, 2)
    conf = ((xy[:, 0] != 0) | (xy[:, 1] != 0)).astype(np.float32)
    return np.concatenate([xy, conf[:, None]], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="e.g. COM5 (Windows) or /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--profile", choices=["desk", "practitioner"], default="desk")
    ap.add_argument("--session", default=None)
    args = ap.parse_args()

    import cv2
    import serial

    session = args.session or f"esp_{args.profile}_{datetime.now():%Y%m%d_%H%M%S}"
    writer = SessionWriter(DATA_DIR / f"{session}.csv")
    ser = serial.Serial(args.port, args.baud, timeout=0.05)
    label, counts, t0 = None, {0: 0, 1: 0}, time.time()
    canvas = np.zeros((140, 640, 3), np.uint8)
    last_info = "waiting for the board..."
    try:
        while True:
            raw = ser.readline().decode("utf-8", "ignore")
            if raw.startswith("KP,"):
                kps = parse_kp(raw)
                if kps is not None and features.person_present(kps):
                    f = features.compute(kps)
                    if label is not None:
                        writer.write(session, time.time() - t0, args.profile, label, kps, f)
                        counts[label] += 1
                    last_info = f"neck {f[0]:5.1f}  trunk {f[1]:5.1f}  neck_rel {f[2]:5.1f}  rules p={rules.score(f):.2f}"
            elif raw.startswith("ST,"):
                pass
            canvas[:] = 30
            status = {None: "PAUSE", 0: "REC GOOD", 1: "REC BAD"}[label]
            put(canvas, f"{status}  good={counts[0]} bad={counts[1]}  {session}", 30)
            put(canvas, last_info, 65)
            put(canvas, "g=good  b=bad  space=pause  q=quit", 120, (200, 200, 200), 0.5)
            cv2.imshow("BAD POSTURE - ESP32 collect", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            label = {ord("g"): 0, ord("b"): 1, ord(" "): None}.get(key, label)
    finally:
        writer.close()
        ser.close()
        cv2.destroyAllWindows()
    print(f"{counts[0]} good, {counts[1]} bad -> {DATA_DIR / (session + '.csv')}")


if __name__ == "__main__":
    main()
