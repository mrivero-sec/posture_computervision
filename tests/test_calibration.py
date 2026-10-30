import io
import math
import struct

import numpy as np
import pytest

from posture import calibration as calib
from posture import features
from posture.keypoints import L_EAR, L_EYE, L_SH, NOSE, R_SH, empty_kps
from posture.serialcam import MAGIC, read_jpeg


def upper_body(ear=(300.0, 200.0), sh=(290.0, 360.0), conf_r_sh=0.2):
    """Left profile, hips out of frame, facing increasing x."""
    k = empty_kps()
    k[L_EAR] = (*ear, 0.9)
    k[L_SH] = (*sh, 0.9)
    k[R_SH] = (sh[0] + 150, sh[1] - 60, conf_r_sh)   # made-up hidden shoulder (low conf)
    k[NOSE] = (ear[0] + 60, ear[1] + 15, 0.9)
    k[L_EYE] = (ear[0] + 45, ear[1] - 5, 0.9)
    return k


def calibrate(kps, n=20):
    c = calib.Calibrator(duration_s=2.0, countdown_s=1.0, min_frames=5)
    c.start(0.0)
    b = err = None
    for i in range(n + 5):
        t = 0.2 * i
        b, err = c.update(t, kps, features.compute(kps))
        if b is not None or err:
            break
    return b, err


def test_calibration_records_baseline():
    k = upper_body()
    b, err = calibrate(k)
    assert err == "" and b is not None
    assert b.ear == pytest.approx((300.0, 200.0))
    assert b.ref == pytest.approx(math.hypot(10, 160))
    assert b.dirx == 1.0


def test_same_posture_is_good():
    k = upper_body()
    b, _ = calibrate(k)
    dev = calib.deviations(b, k, features.compute(k))
    assert all(abs(dev[n]) < 1e-6 for n in ("head_drop", "head_fwd", "sh_drop", "sh_fwd", "d_neck"))
    assert calib.score(dev) < 0.05


def test_slump_is_bad_without_hips():
    b, _ = calibrate(upper_body())
    # slumped: head and shoulder drop and move forward by ~70 px (~0.45 ref)
    k = upper_body(ear=(370.0, 270.0), sh=(330.0, 410.0))
    f = features.compute(k)
    assert f[features.FEATURE_NAMES.index("hips_visible")] == 0.0
    dev = calib.deviations(b, k, f)
    assert dev["head_drop"] > 0.35 and dev["head_fwd"] > 0.35
    assert calib.score(dev) > 0.6


def test_scale_change_invalidates():
    base = upper_body()
    b, _ = calibrate(base)
    k = base.copy()
    k[:, :2] *= 1.6                                          # everything 1.6x larger: moved closer
    assert calib.score(calib.deviations(b, k, features.compute(k))) is None


def test_compressed_neck_is_bad_not_stale():
    """Slumped: ear-shoulder 30 % shorter, shoulder lower, same head size."""
    b, _ = calibrate(upper_body())
    k = upper_body(ear=(300.0, 268.0), sh=(290.0, 380.0))
    dev = calib.deviations(b, k, features.compute(k))
    assert 0.75 <= dev["scale"] <= 1.33          # not mistaken for moving back
    assert dev["neck_short"] > 0.25
    assert calib.score(dev) > 0.9


def test_looking_down_alone_is_ok_in_trunk_mode():
    """Looking down / writing: the head moves, the shoulders stay put."""
    b, _ = calibrate(upper_body())
    k = upper_body(ear=(330.0, 250.0), sh=(290.0, 360.0))
    dev = calib.deviations(b, k, features.compute(k))
    assert dev["head_drop"] > 0.25 and abs(dev["sh_drop"]) < 1e-6
    assert calib.score(dev) < 0.05                       # trunk first: no alert
    assert calib.score(dev, mode="all") > 0.5            # previous behaviour: alert


def test_small_body_rotation_is_ok():
    """Chair swivel of about ten degrees: shoulder shifted horizontally by ~0.08 ref."""
    b, _ = calibrate(upper_body())
    k = upper_body(ear=(312.0, 200.0), sh=(303.0, 360.0))
    assert calib.score(calib.deviations(b, k, features.compute(k))) < 0.1


def test_old_baseline_without_head_size_still_works():
    b, _ = calibrate(upper_body())
    b.head_nose = b.head_eye = 0.0
    k = upper_body()
    assert calib.score(calib.deviations(b, k, features.compute(k))) < 0.05


def test_too_few_frames_fails():
    c = calib.Calibrator(duration_s=1.0, countdown_s=0.0, min_frames=5)
    c.start(0.0)
    b, err = c.update(0.5, None, None)
    b, err = c.update(1.1, None, None)
    assert b is None and "failed" in err


def test_baseline_roundtrip(tmp_path):
    b, _ = calibrate(upper_body())
    p = tmp_path / "c.json"
    b.save(p)
    b2 = calib.Baseline.load(p)
    assert b2.ear == b.ear and b2.features == b.features


class FakeSerial:
    def __init__(self, payload: bytes):
        self.buf = io.BytesIO(payload)
        self.written = b""

    def reset_input_buffer(self):
        pass

    def write(self, b):
        self.written += b

    def read(self, n):
        return self.buf.read(n)


def test_serial_protocol_skips_text_lines():
    jpg = b"\xff\xd8fakejpeg\xff\xd9"
    logs = []
    ser = FakeSerial(b"OK camera PID=0x26\n" + MAGIC + struct.pack("<I", len(jpg)) + jpg)
    assert read_jpeg(ser, log=logs.append) == jpg
    assert ser.written == b"c"
    assert logs == ["[board] OK camera PID=0x26"]


def test_serial_protocol_truncated():
    ser = FakeSerial(MAGIC + struct.pack("<I", 100) + b"abc")
    assert read_jpeg(ser, timeout=0.2, log=None) is None


def facing_left(ear=(300.0, 200.0), sh=(310.0, 360.0)):
    """Profile facing decreasing x (like the reference setup)."""
    k = empty_kps()
    k[L_EAR] = (*ear, 0.9)
    k[L_SH] = (*sh, 0.9)
    k[NOSE] = (ear[0] - 60, ear[1] + 15, 0.9)
    k[L_EYE] = (ear[0] - 45, ear[1] - 5, 0.9)
    return k


def hide(k, *idx):
    k = k.copy()
    for i in idx:
        k[i] = (0.0, 0.0, 0.0)
    return k


NECK = features.FEATURE_NAMES.index("neck_incl")


def test_hand_over_mouth_keeps_neck_angle():
    bad = facing_left(ear=(230.0, 230.0), sh=(310.0, 360.0))   # head far forward
    ref = features.compute(bad)[NECK]
    assert ref > 25
    # nose hidden: the eyes give the direction
    assert features.compute(hide(bad, NOSE))[NECK] == pytest.approx(ref)
    # nose and eyes hidden: head assumed ahead of the shoulder
    assert features.compute(hide(bad, NOSE, L_EYE))[NECK] == pytest.approx(ref)


def test_calibration_ignores_hidden_nose():
    b, _ = calibrate(facing_left())
    bad = facing_left(ear=(230.0, 260.0), sh=(290.0, 400.0))
    s_full = calib.score(calib.deviations(b, bad, features.compute(bad)))
    covered = hide(bad, NOSE, L_EYE)
    s_cov = calib.score(calib.deviations(b, covered, features.compute(covered)))
    assert s_full > 0.6 and s_cov == pytest.approx(s_full)
