"""Personal calibration: reference position in good posture.

The camera is fixed. A few seconds of good posture are recorded, then the
deviation from that reference is measured in units of the reference
"ear-shoulder length" (about 15-20 cm for an adult). This detects slouching
and leaning forward even when the hips are out of frame.

Deviations (positive = towards bad posture):
  d_neck     ear-shoulder inclination beyond the reference (deg)
  head_drop  ear lower than the reference (back sagging)
  head_fwd   ear further forward than the reference (forward head)
  sh_drop    shoulder lower than the reference (rounded back)
  sh_fwd     shoulder further forward than the reference (trunk leaning)
  neck_short shorter ear-shoulder distance (head sinking, rounded back)

If the person has moved closer to or further from the camera, the calibration
is considered stale. Scale = sqrt(head size x ear-shoulder length), each
relative to the reference. Changing the distance to the camera makes both
bigger (or smaller) together; bad posture moves them in opposite directions
(shorter neck, head looking bigger when leaning in), so the product stays
close to 1. Observed on the 2026-10-29 data: either one alone fell outside
[0.75, 1.33] in 40 to 60 % of the "bad" frames.

Mirrored in firmware/main/calibration.cpp (tests/test_parity.py).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES
from .keypoints import L_EAR, L_EYE, L_SH, NOSE, R_EAR, R_EYE, R_SH

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
CALIB_DIR = Path(__file__).resolve().parent.parent / "data"

DEFAULT_THRESHOLDS = {      # (threshold, scale) as in rules.py; score 0.5 at the threshold
    "sh_drop": (0.12, 0.07),    # trunk: shoulder drops (back sagging)
    "sh_fwd": (0.30, 0.10),     # trunk: shoulder moves forward (leaning in)
    "d_neck": (12.0, 5.0),      # head
    "head_drop": (0.35, 0.15),
    "head_fwd": (0.45, 0.15),
    "neck_short": (0.15, 0.07),
}
TRUNK_KEYS = ("sh_drop", "sh_fwd")
HEAD_KEYS = ("d_neck", "head_drop", "head_fwd", "neck_short")
DEFAULTS = {"duration_s": 5.0, "countdown_s": 3.0, "min_frames": 8, "scale_min": 0.75, "scale_max": 1.33,
            # "trunk" mode: the head only counts if the shoulders have moved too.
            # Looking down (shoulders still) therefore triggers nothing.
            "mode": "trunk", "gate_sh_drop": 0.07, "gate_sh_fwd": 0.10}


def side_points(kps, thr: float = 0.5):
    """Ear and shoulder on the best-seen side + facing direction. None if missing."""
    def conf(i):
        return float(kps[i][2]) if float(kps[i][2]) >= thr else 0.0

    left = conf(L_EAR) + conf(L_SH)
    right = conf(R_EAR) + conf(R_SH)
    ear_i, sh_i = (L_EAR, L_SH) if left >= right else (R_EAR, R_SH)
    if conf(ear_i) == 0.0:
        ear_i = R_EAR if ear_i == L_EAR else L_EAR
    if conf(sh_i) == 0.0:
        sh_i = R_SH if sh_i == L_SH else L_SH
    if conf(ear_i) == 0.0 or conf(sh_i) == 0.0:
        return None
    ear = (float(kps[ear_i][0]), float(kps[ear_i][1]))
    sh = (float(kps[sh_i][0]), float(kps[sh_i][1]))
    dirx = 0.0
    for i in (NOSE, L_EYE, R_EYE):       # nose, otherwise an eye
        if conf(i) > 0.0 and abs(float(kps[i][0]) - ear[0]) > 1e-6:
            dirx = 1.0 if float(kps[i][0]) > ear[0] else -1.0
            break
    return ear, sh, dirx


def head_sizes(kps, ear, thr: float = 0.5) -> tuple[float, float]:
    """(ear-nose, ear-eye) in px, 0 if not visible. Independent of posture."""
    dn = de = 0.0
    if float(kps[NOSE][2]) >= thr:
        dn = math.hypot(float(kps[NOSE][0]) - ear[0], float(kps[NOSE][1]) - ear[1])
    eye = max((L_EYE, R_EYE), key=lambda i: float(kps[i][2]))
    if float(kps[eye][2]) >= thr:
        de = math.hypot(float(kps[eye][0]) - ear[0], float(kps[eye][1]) - ear[1])
    return dn, de


@dataclass
class Baseline:
    ear: tuple
    sh: tuple
    dirx: float
    ref: float              # ear-shoulder distance (px)
    features: list           # median features in good posture
    created: str = ""
    profile: str = ""
    n_frames: int = 0
    head_nose: float = 0.0   # reference ear-nose distance (px), for the scale
    head_eye: float = 0.0    # reference ear-eye distance (px)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Baseline | None":
        if not Path(path).exists():
            return None
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["ear"], d["sh"] = tuple(d["ear"]), tuple(d["sh"])
        return cls(**d)


def default_path(profile: str) -> Path:
    return CALIB_DIR / f"calibration_{profile}.json"


def _neck_angle(ear, sh, dirx: float) -> float:
    """Shoulder->ear inclination vs vertical, + = head forward (deg)."""
    return math.degrees(math.atan2(dirx * (ear[0] - sh[0]), -(ear[1] - sh[1])))


def deviations(b: Baseline, kps, f, thr: float = 0.5) -> dict | None:
    """Deviations from the reference, or None if they cannot be measured."""
    sp = side_points(kps, thr)
    if sp is None or b.ref <= 1e-6:
        return None
    ear, sh, _ = sp
    dn, de = head_sizes(kps, ear, thr)
    neck = math.hypot(ear[0] - sh[0], ear[1] - sh[1]) / b.ref
    if de > 0 and b.head_eye > 0:
        scale = math.sqrt(de / b.head_eye * neck)
    elif dn > 0 and b.head_nose > 0:
        scale = math.sqrt(dn / b.head_nose * neck)
    else:
        scale = 1.0          # head size unknown: trust the calibration
    # fixed camera, person always facing the same way: keep the reference
    # direction (robust when the nose is hidden by a hand)
    dirx = b.dirx or 1.0
    return {
        "d_neck": _neck_angle(ear, sh, dirx) - _neck_angle(b.ear, b.sh, dirx),
        "head_drop": (ear[1] - b.ear[1]) / b.ref,
        "head_fwd": dirx * (ear[0] - b.ear[0]) / b.ref,
        "sh_drop": (sh[1] - b.sh[1]) / b.ref,
        "sh_fwd": dirx * (sh[0] - b.sh[0]) / b.ref,
        "neck_short": 1.0 - neck,
        "scale": scale,
    }


def trunk_moved(dev: dict, gate_sh_drop=0.07, gate_sh_fwd=0.10) -> bool:
    return dev.get("sh_drop", 0.0) > gate_sh_drop or dev.get("sh_fwd", 0.0) > gate_sh_fwd


def exceedances(dev: dict, thresholds=None, mode="trunk", gate_sh_drop=0.07, gate_sh_fwd=0.10) -> dict:
    t = thresholds or DEFAULT_THRESHOLDS
    head_ok = mode != "trunk" or trunk_moved(dev, gate_sh_drop, gate_sh_fwd)
    return {k: (dev[k] - thr) / sc for k, (thr, sc) in t.items()
            if k in dev and (head_ok or k not in HEAD_KEYS)}


def score(dev: dict | None, thresholds=None, scale_min=0.75, scale_max=1.33,
          mode="trunk", gate_sh_drop=0.07, gate_sh_fwd=0.10) -> float | None:
    """Bad posture probability from the deviations; None if not applicable."""
    if dev is None or not (scale_min <= dev["scale"] <= scale_max):
        return None
    ex = exceedances(dev, thresholds, mode, gate_sh_drop, gate_sh_fwd)
    return 1.0 / (1.0 + math.exp(-2.0 * max(ex.values()))) if ex else None


def worst(dev: dict | None, thresholds=None, mode="trunk", gate_sh_drop=0.07, gate_sh_fwd=0.10) -> str:
    if dev is None:
        return ""
    ex = exceedances(dev, thresholds, mode, gate_sh_drop, gate_sh_fwd)
    return max(ex, key=ex.get) if ex else ""


def score_kw(params: dict) -> dict:
    """score() / worst() arguments taken from params_from_cfg()."""
    return {k: params[k] for k in ("mode", "gate_sh_drop", "gate_sh_fwd")}


@dataclass
class Calibrator:
    """State machine: IDLE -> COUNTDOWN -> RECORDING -> IDLE (+ baseline)."""
    profile: str = "desk"
    duration_s: float = DEFAULTS["duration_s"]
    countdown_s: float = DEFAULTS["countdown_s"]
    min_frames: int = DEFAULTS["min_frames"]
    state: str = "IDLE"
    t_start: float = 0.0
    _pts: list = field(default_factory=list)
    _feats: list = field(default_factory=list)
    _heads: list = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.state != "IDLE"

    def start(self, now: float) -> None:
        self.state, self.t_start = "COUNTDOWN", now
        self._pts, self._feats, self._heads = [], [], []

    def message(self, now: float) -> str:
        if self.state == "COUNTDOWN":
            return f"CALIBRATION: sit up straight... {math.ceil(self.countdown_s - (now - self.t_start))}"
        if self.state == "RECORDING":
            left = self.duration_s - (now - self.t_start - self.countdown_s)
            return f"CALIBRATION: hold still ({max(left, 0):.0f} s)  {len(self._pts)} frames"
        return ""

    def update(self, now: float, kps, f, thr: float = 0.5) -> tuple[Baseline | None, str]:
        """Call on every frame. Returns (baseline, error) at the end, otherwise (None, '')."""
        if self.state == "COUNTDOWN" and now - self.t_start >= self.countdown_s:
            self.state = "RECORDING"
        if self.state == "RECORDING":
            sp = side_points(kps, thr) if kps is not None else None
            if sp is not None and f is not None:
                self._pts.append(sp)
                self._heads.append(head_sizes(kps, sp[0], thr))
                self._feats.append(np.asarray(f, dtype=np.float64))
            if now - self.t_start >= self.countdown_s + self.duration_s:
                self.state = "IDLE"
                if len(self._pts) < self.min_frames:
                    return None, f"calibration failed: only {len(self._pts)} usable frames"
                ear = np.median([p[0] for p in self._pts], axis=0)
                sh = np.median([p[1] for p in self._pts], axis=0)
                dirs = [p[2] for p in self._pts if p[2] != 0.0]
                dirx = 1.0 if not dirs or np.median(dirs) >= 0 else -1.0
                ref = float(np.median([math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]) for p in self._pts]))
                b = Baseline(
                    ear=(float(ear[0]), float(ear[1])), sh=(float(sh[0]), float(sh[1])), dirx=dirx, ref=ref,
                    features=[float(v) for v in np.median(np.stack(self._feats), axis=0)],
                    created=time.strftime("%Y-%m-%d %H:%M:%S"), profile=self.profile, n_frames=len(self._pts),
                    head_nose=_median_pos([h[0] for h in self._heads]),
                    head_eye=_median_pos([h[1] for h in self._heads]),
                )
                return b, ""
        return None, ""


def _median_pos(v) -> float:
    v = [x for x in v if x > 0]
    return float(np.median(v)) if v else 0.0


def thresholds_from_cfg(cfg: dict) -> dict:
    t = (cfg.get("calibration") or {}).get("thresholds") or {}
    return {k: tuple(v) for k, v in t.items()} or DEFAULT_THRESHOLDS


def params_from_cfg(cfg: dict) -> dict:
    c = dict(DEFAULTS)
    c.update({k: v for k, v in (cfg.get("calibration") or {}).items() if k != "thresholds"})
    return c
