"""Shows the ESP32-S3 board camera over the USB cable, without Wi-Fi.

Use with the arduino/cam_serial/cam_serial.ino sketch.

  python scripts/serial_viewer.py --port COM3
  python scripts/serial_viewer.py --port COM3 --pose --rotate 270   # + YOLO11n-pose skeleton, board in portrait

Keys (window focused): s = save the frame to data/captures, q = quit.
Close the Arduino IDE Serial Monitor before starting (one program per port).
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import _common  # noqa: F401
from _common import ROOT

import cv2

from posture.serialcam import Rotated, SerialCam

_gui = cv2.getBuildInformation().split("GUI:")
if len(_gui) > 1 and "NONE" in _gui[1].splitlines()[0]:
    raise SystemExit(
        "OpenCV is installed without window support (opencv-python-headless).\n"
        "Fix it with:  pip uninstall -y opencv-python-headless opencv-python  then  pip install opencv-python"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="e.g. COM3")
    ap.add_argument("--baud", type=int, default=2_000_000)
    ap.add_argument("--pose", action="store_true", help="draw the YOLO11n-pose skeleton (computed on the PC)")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                    help="clockwise frame rotation, e.g. 270 if the board stands in portrait")
    args = ap.parse_args()

    model = None
    if args.pose:
        from posture.keypoints import load_yolo
        model = load_yolo("yolo11n-pose.pt")

    raw = SerialCam(args.port, args.baud)
    cap = Rotated(raw, args.rotate) if args.rotate else raw
    out_dir = ROOT / "data" / "captures"
    fps, t_last = 0.0, time.time()
    print("Connected. s = save, q = quit.")
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                continue
            now = time.time()
            fps = 0.8 * fps + 0.2 / max(now - t_last, 1e-3)
            t_last = now
            shown = model(img, verbose=False)[0].plot() if model else img.copy()
            cv2.putText(shown, f"{img.shape[1]}x{img.shape[0]}  {len(raw.last_jpeg)//1024} KB  {fps:.1f} fps",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("BAD POSTURE - USB camera", shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out_dir.mkdir(parents=True, exist_ok=True)
                p = out_dir / f"cap_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(p), img)
                print("Saved:", p)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
