"""Real-time detection + alerts (simulates the LED and buzzer on the PC).

Keys: c = calibrate (good posture for 5 s), s = snooze 15 min, p = switch profile, q = quit,
      f = flag a false alert, m = flag a missed alert (notes in the log).

A CSV log (numbers only, no images) is written to data/logs/ by default.

Examples:
  python scripts/run.py                      # source and rotation from config.yaml (board camera)
  python scripts/run.py --calibrate          # start with a calibration
  python scripts/run.py --source 0 --rotate 0   # PC webcam
  python scripts/run.py --profile practitioner --no-sound
  python scripts/run.py --source video.mp4 --log run_log.csv
"""
from __future__ import annotations

import argparse
import csv
import time

import _common  # noqa: F401
from _common import open_source, put

from posture import calibration as calib
from posture import config, features, rules
from posture.alert import AlertParams, AlertState
from posture.feedback import STATE_COLOR, beep
from posture.keypoints import draw, make_backend
from posture.model import PostureModel


def main():
    cfg = config.load()
    cam = cfg["camera"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=list(cfg["profiles"]), default="desk")
    ap.add_argument("--source", default=str(cam.get("source", cam["index"])),
                    help="webcam index, video file or serial:COM3")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=int(cam.get("rotate", 0)))
    ap.add_argument("--backend", default=cfg["pose"]["backend"])
    ap.add_argument("--model", default=None, help="path to the .joblib (default models/posture_clf.joblib, otherwise rules)")
    ap.add_argument("--calibrate", action="store_true", help="calibrate at startup")
    ap.add_argument("--no-calib", action="store_true", help="ignore the saved calibration")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--log", default=None, help="log CSV (default data/logs/run_<date>.csv)")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--headless", action="store_true", help="no window (log only)")
    args = ap.parse_args()

    import cv2

    profiles = list(cfg["profiles"])
    prof = args.profile
    model = PostureModel(args.model, cfg["rules"])
    print(f"Classifier: {model.kind}" + ("" if cfg.get("calibration", {}).get("use_model_with_calib", False)
                                           else " (replaced by the calibration once calibrated)"))
    cal_thr = calib.thresholds_from_cfg(cfg)
    cal_par = calib.params_from_cfg(cfg)
    cal_kw = calib.score_kw(cal_par)

    def make_state(name):
        return AlertState(AlertParams.from_dict(cfg["profiles"][name]))

    def load_baseline(name):
        if args.no_calib:
            return None
        b = calib.Baseline.load(calib.default_path(name))
        if b:
            print(f"Calibration '{name}' loaded ({b.created}, {b.n_frames} frames). Press c to redo it.")
        return b

    def make_calibrator(name):
        return calib.Calibrator(profile=name, duration_s=cal_par["duration_s"],
                                countdown_s=cal_par["countdown_s"], min_frames=cal_par["min_frames"])

    st = make_state(prof)
    baseline = load_baseline(prof)
    calibrator = make_calibrator(prof)
    kw = {"weights": cfg["pose"]["weights"], "imgsz": cfg["pose"]["imgsz"]} if args.backend == "yolo" else {}
    pose = make_backend(args.backend, **kw)
    cap = open_source(args.source, cam["width"], cam["height"], args.rotate)
    thr = cfg["pose"]["conf_thr"]
    period = 1.0 / max(cam["fps_limit"], 0.1)
    is_file = not args.source.isdigit() and not args.source.lower().startswith("serial:")
    fps_file = (cap.get(cv2.CAP_PROP_FPS) if is_file else 0.0) or 25.0
    log_fh = None
    if not args.no_log:
        from datetime import datetime
        from pathlib import Path

        log_path = Path(args.log) if args.log else _common.ROOT / "data" / "logs" / f"run_{datetime.now():%Y%m%d_%H%M%S}.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", newline="", encoding="utf-8")
        print(f"Log: {log_path}  (f = false alert, m = missed alert)")
    log = csv.writer(log_fh) if log_fh else None
    DEV_COLS = ["sh_drop", "sh_fwd", "d_neck", "head_drop", "head_fwd", "neck_short", "scale"]
    if log:
        log.writerow(["t", "prob", "model_prob", "calib_prob", "ema", "state", "worst", *DEV_COLS, "event"])
    event = ""

    t0 = time.time()
    frame_i = 0
    last = -1e9
    info = ""
    if args.calibrate:
        calibrator.start(0.0 if is_file else time.time() - t0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        now = frame_i / fps_file if is_file else time.time() - t0  # video time for files
        if now - last < period:
            if not is_file and not args.headless:
                cv2.waitKey(1)
            continue
        last = now
        kps = pose(frame)
        present = features.person_present(kps, thr)
        f = features.compute(kps, thr) if present else None

        prob = cprob = mprob = dev = None
        worst = ""
        if calibrator.active:
            b, err = calibrator.update(now, kps if present else None, f, thr)
            if b is not None:
                baseline = b
                b.save(calib.default_path(prof))
                info = f"Calibration saved ({b.n_frames} frames)"
                print(info)
            elif err:
                info = err
                print(err)
            state, do_beep = st.state, False
        else:
            if f is not None:
                prob = mprob = model.predict_proba(f)
                worst = rules.worst(f, cfg["rules"])
                if baseline is not None:
                    dev = calib.deviations(baseline, kps, f, thr)
                    cprob = calib.score(dev, cal_thr, cal_par["scale_min"], cal_par["scale_max"], **cal_kw)
                    if cprob is not None and (not cal_par["use_model_with_calib"] or cprob > prob):
                        prob, worst = cprob, "calib:" + calib.worst(dev, cal_thr, **cal_kw)
            state, do_beep = st.update(now, prob)
        sound_on = cfg["profiles"][prof].get("sound", True) and not args.no_sound
        if do_beep and sound_on:
            beep()
        if log:
            fmt = lambda v: "" if v is None else f"{v:.3f}"  # noqa: E731
            log.writerow([f"{now:.2f}", fmt(prob), fmt(mprob), fmt(cprob), f"{st.ema:.3f}", state, worst,
                          *[fmt(dev[k]) if dev else "" for k in DEV_COLS], event])
            event = ""

        if args.headless:
            continue
        color = STATE_COLOR[state]
        draw(frame, kps, thr)
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 12)
        src = "calibration" if (cprob is not None and not cal_par["use_model_with_calib"]) else model.kind
        put(frame, f"{state}  conf={st.ema:.2f}  {prof}  {src}", 28, color, 0.6)
        if calibrator.active:
            put(frame, calibrator.message(now), 56, (0, 255, 255), 0.6)
        elif f is not None:
            put(frame, f"neck {f[0]:5.1f}  trunk {f[1]:5.1f}  worst: {worst}", 54)
            if baseline is None:
                put(frame, "not calibrated: press c to calibrate", 76, (0, 200, 255))
            elif dev is not None and cprob is None:
                put(frame, f"calibration stale (scale {dev['scale']:.2f}): press c to redo it", 76, (0, 200, 255))
            elif dev is not None:
                put(frame, f"d_neck {dev['d_neck']:+5.1f}  head_drop {dev['head_drop']:+.2f}  head_fwd {dev['head_fwd']:+.2f}", 76)
                put(frame, f"sh_drop {dev['sh_drop']:+.2f}  sh_fwd {dev['sh_fwd']:+.2f}  neck_short {dev['neck_short']:+.2f}  p={cprob:.2f}", 98)
        if info and not calibrator.active:
            put(frame, info, frame.shape[0] - 36, (0, 255, 255), 0.5)
        put(frame, "c=calibrate s=snooze p=profile f=false alert m=missed q=quit", frame.shape[0] - 14, (220, 220, 220), 0.42)
        cv2.imshow("BAD POSTURE", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            info = ""
            calibrator.start(now)
        if key in (ord("f"), ord("m")):
            event = "FP" if key == ord("f") else "MISS"
            info = f"noted: {'false alert' if event == 'FP' else 'missed alert'} at {now / 60:.1f} min"
            print(info)
        if key == ord("s"):
            st.snooze(now)
        if key == ord("p"):
            prof = profiles[(profiles.index(prof) + 1) % len(profiles)]
            st = make_state(prof)
            baseline = load_baseline(prof)
            calibrator = make_calibrator(prof)
    cap.release()
    if log_fh:
        log_fh.close()
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
