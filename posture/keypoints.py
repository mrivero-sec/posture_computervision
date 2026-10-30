"""Pose estimation -> 17 keypoints in COCO format.

COCO-17 is used everywhere (PC and ESP32-S3) so that a classifier trained on
the PC can be deployed on the board as is (Espressif's coco_pose YOLO11n-pose
model).

Common output: float32 ndarray of shape (17, 3) -> (x_pixel, y_pixel, conf).
A missing keypoint is (0, 0, 0), as in ESP-DL.
"""
from __future__ import annotations

import numpy as np

COCO_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_ELB, R_ELB, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP = 11, 12

# Segments drawn on screen
SKELETON = [
    (L_EAR, L_EYE), (L_EYE, NOSE), (NOSE, R_EYE), (R_EYE, R_EAR),
    (L_SH, R_SH), (L_SH, L_ELB), (L_ELB, L_WR), (R_SH, R_ELB), (R_ELB, R_WR),
    (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP),
    (L_EAR, L_SH), (R_EAR, R_SH),
]

# MediaPipe Pose (33 points) -> COCO-17
_MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def empty_kps() -> np.ndarray:
    return np.zeros((17, 3), dtype=np.float32)


def load_yolo(weights: str = "yolo11n-pose.pt"):
    """Load YOLO with Ultralytics usage analytics turned off."""
    try:
        from ultralytics import settings

        if settings.get("sync", False):
            settings.update({"sync": False})
    except Exception:  # older versions without settings: carry on
        pass
    from ultralytics import YOLO  # lazy import: heavy dependency

    return YOLO(weights)


class YoloPose:
    """Ultralytics YOLO11n-pose: same family as the ESP-DL coco_pose model."""

    def __init__(self, weights: str = "yolo11n-pose.pt", imgsz: int = 320, device: str | None = None):
        self.model = load_yolo(weights)
        self.imgsz = imgsz
        self.device = device

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        res = self.model.predict(frame_bgr, imgsz=self.imgsz, verbose=False, device=self.device)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return empty_kps()
        # largest person = closest to the camera
        boxes = res.boxes.xywh.cpu().numpy()
        idx = int(np.argmax(boxes[:, 2] * boxes[:, 3]))
        xy = res.keypoints.xy[idx].cpu().numpy()
        conf = res.keypoints.conf
        conf = conf[idx].cpu().numpy() if conf is not None else np.ones(17)
        out = np.concatenate([xy, conf[:, None]], axis=1).astype(np.float32)
        out[(out[:, 0] == 0) & (out[:, 1] == 0), 2] = 0.0
        return out


class MediaPipePose:
    """Lightweight alternative (Apache 2.0 licence), remapped to COCO-17."""

    def __init__(self, model_complexity: int = 1):
        import mediapipe as mp

        self._mp = mp
        self.pose = mp.solutions.pose.Pose(model_complexity=model_complexity)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        h, w = frame_bgr.shape[:2]
        res = self.pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return empty_kps()
        lm = res.pose_landmarks.landmark
        out = empty_kps()
        for coco_i, mp_i in enumerate(_MP_TO_COCO):
            p = lm[mp_i]
            out[coco_i] = (p.x * w, p.y * h, p.visibility)
        return out


def make_backend(name: str, **kw):
    name = name.lower()
    if name in ("yolo", "ultralytics"):
        return YoloPose(**kw)
    if name in ("mediapipe", "mp"):
        return MediaPipePose(**kw)
    raise ValueError(f"unknown backend: {name}")


def draw(frame, kps: np.ndarray, conf_thr: float = 0.5, color=(0, 255, 0)):
    import cv2

    for a, b in SKELETON:
        if kps[a, 2] >= conf_thr and kps[b, 2] >= conf_thr:
            cv2.line(frame, tuple(int(v) for v in kps[a, :2]), tuple(int(v) for v in kps[b, :2]), color, 2)
    for x, y, c in kps:
        if c >= conf_thr:
            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)
    return frame
