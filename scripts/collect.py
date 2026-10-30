"""Labelled data collection (no images are saved).

Default source: config.yaml (board camera over USB, rotation included).

Keys:
  c      calibrate (good posture for 5 s) -> data/<session>.calib.json
  g      record GOOD posture (label 0)
  b      record BAD posture (label 1)
  space  pause (records nothing)
  q      quit

Examples:
  python scripts/collect.py --profile desk --session desk_01
  python scripts/collect.py --source 0 --rotate 0      # PC webcam
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import _common  # noqa: F401  (adds the repo root to sys.path)
from _common import open_source, put

from posture import calibration as calib
from posture import config, features, rules
from posture.io import DATA_DIR, SessionWriter
from posture.keypoints import draw, make_backend


def main():
    cfg = config.load()
    cam = cfg["camera"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=list(cfg["profiles"]), default="desk")
    ap.add_argument("--session", default=None, help="session name (default: date/time)")
    ap.add_argument("--source", default=str(cam.get("source", cam["index"])),
                    help="webcam index, video file or serial:COM3")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=int(cam.get("rotate", 0)))
    ap.add_argument("--backend", default=cfg["pose"]["backend"])
    args = ap.parse_args()

    session = args.session or f"{args.profile}_{datetime.now():%Y%m%d_%H%M%S}"
    out = DATA_DIR / f"{session}.csv"
    calib_out = DATA_DIR / f"{session}.calib.json"
    writer = SessionWriter(out)

    import cv2

    kw = {"weights": cfg["pose"]["weights"], "imgsz": cfg["pose"]["imgsz"]} if args.backend == "yolo" else {}
    pose = make_backend(args.backend, **kw)
    cap = open_source(args.source, cam["width"], cam["height"], args.rotate)
    thr = cfg["pose"]["conf_thr"]
    period = 1.0 / max(cam["fps_limit"], 0.1)
    cal_thr = calib.thresholds_from_cfg(cfg)
    cal_par = calib.params_from_cfg(cfg)
    calibrator = calib.Calibrator(profile=args.profile, duration_s=cal_par["duration_s"],
                                  countdown_s=cal_par["countdown_s"], min_frames=cal_par["min_frames"])
    baseline = calib.Baseline.load(calib_out)

    label = None  # None = pause, 0 = good, 1 = bad
    counts = {0: 0, 1: 0}
    t0 = time.time()
    last = 0.0
    info = "" if baseline else "start with c (calibration), then g / b"
    print(f"Session {session} -> {out}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            if now - last < period:
                key = cv2.waitKey(1) & 0xFF
            else:
                last = now
                kps = pose(frame)
                present = features.person_present(kps, thr)
                f = features.compute(kps, thr)
                if calibrator.active:
                    b, err = calibrator.update(now - t0, kps if present else None, f if present else None, thr)
                    if b is not None:
                        baseline = b
                        b.save(calib_out)
                        b.save(calib.default_path(args.profile))   # reused by run.py
                        info = f"calibration saved ({b.n_frames} frames)"
                    elif err:
                        info = err
                elif present and label is not None:
                    writer.write(session, now - t0, args.profile, label, kps, f)
                    counts[label] += 1
                draw(frame, kps, thr)
                status = {None: ("PAUSE", (200, 200, 200)), 0: ("REC GOOD", (0, 220, 0)), 1: ("REC BAD", (0, 0, 255))}[label]
                put(frame, f"{status[0]}  good={counts[0]} bad={counts[1]}  {args.profile}", 22, status[1])
                if calibrator.active:
                    put(frame, calibrator.message(now - t0), 46, (0, 255, 255), 0.6)
                elif present:
                    put(frame, f"neck {f[0]:5.1f}  trunk {f[1]:5.1f}  rules p={rules.score(f, cfg['rules']):.2f}", 46)
                    dev = calib.deviations(baseline, kps, f, thr) if baseline else None
                    if dev is not None:
                        cp = calib.score(dev, cal_thr, cal_par["scale_min"], cal_par["scale_max"], **calib.score_kw(cal_par))
                        put(frame, f"calib: d_neck {dev['d_neck']:+5.1f}  head_drop {dev['head_drop']:+.2f} "
                                   f"head_fwd {dev['head_fwd']:+.2f}  p={'-' if cp is None else f'{cp:.2f}'}", 68)
                else:
                    put(frame, "no person detected", 46, (0, 165, 255))
                if info:
                    put(frame, info, frame.shape[0] - 34, (0, 255, 255), 0.5)
                put(frame, "c=calibrate g=good b=bad space=pause q=quit", frame.shape[0] - 12, (220, 220, 220), 0.45)
                cv2.imshow("BAD POSTURE - collect", frame)
                key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                label, info = None, ""
                calibrator.start(time.time() - t0)
            elif key == ord("g"):
                label = 0
            elif key == ord("b"):
                label = 1
            elif key == ord(" "):
                label = None
    finally:
        writer.close()
        cap.release()
        cv2.destroyAllWindows()
    print(f"Done: {counts[0]} good frames, {counts[1]} bad frames -> {out}")


if __name__ == "__main__":
    main()
