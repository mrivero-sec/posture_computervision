"""Shared helpers for the scripts (package path, opening the camera)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def open_source(source: str, width: int, height: int, rotate: int = 0):
    """Video source: webcam index ("0"), a file, or the board camera ("serial:COM3")."""
    from posture.serialcam import Rotated

    cap = _open_raw(source, width, height)
    return Rotated(cap, rotate) if rotate else cap


def _open_raw(source: str, width: int, height: int):
    import cv2

    from posture.serialcam import SerialCam

    if source.lower().startswith("serial:"):
        parts = source.split(":")          # serial:COM3 or serial:COM3:2000000
        baud = int(parts[2]) if len(parts) > 2 else 2_000_000
        return SerialCam(parts[1], baud)
    if source.isdigit():
        idx = int(source)
        # CAP_DSHOW avoids a 10 s delay when opening on Windows
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open the video source: {source}")
    return cap


def put(frame, text, y, color=(255, 255, 255), scale=0.55):
    import cv2

    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
