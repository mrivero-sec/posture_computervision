"""Verifie que le firmware C++ (features, regles, classifieur, alertes) donne
exactement les memes resultats que le Python. Necessite g++ (sinon test ignore)."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from posture import features, rules  # noqa: E402
from posture.alert import AlertParams, AlertState  # noqa: E402

FW = ROOT / "firmware" / "main"
pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ absent")


def build(tmp: Path, model_header: Path | None) -> Path:
    exe = tmp / ("parity_model" if model_header else "parity_rules")
    cmd = ["g++", "-std=c++17", "-O1", f"-I{FW}", str(ROOT / "tests" / "host_c" / "parity_main.cpp"),
           str(FW / "features.cpp"), str(FW / "alert.cpp"), str(FW / "calibration.cpp"), "-o", str(exe)]
    if model_header:
        cmd.insert(3, f'-DPOSTURE_MODEL_HEADER="{model_header}"')
    else:
        cmd.insert(3, "-DPOSTURE_FORCE_RULES")
    subprocess.run(cmd, check=True)
    return exe


def random_kps(n=400, seed=1):
    import make_synthetic as ms

    rng = np.random.default_rng(seed)
    ks = [ms.sample(rng, "desk" if i % 2 else "practitioner")[0] for i in range(n)]
    # quelques cas degeneres : points manquants au hasard
    for k in ks[::7]:
        k[rng.integers(0, 17, size=4), :] = 0
    ks.append(np.zeros((17, 3), np.float32))
    return np.stack(ks)


def run_exe(exe, kps, mode="features"):
    inp = "\n".join(",".join(f"{v:.6f}" for v in k.reshape(-1)) for k in kps) + "\n"
    out = subprocess.run([str(exe), mode], input=inp, capture_output=True, text=True, check=True).stdout
    return np.array([[float(v) for v in line.split(",")] for line in out.strip().splitlines()])


def _kps_roundtrip(kps):
    # le harnais lit 6 decimales : on compare sur les memes valeurs
    return np.round(kps.astype(np.float64), 6).astype(np.float32)


def test_features_and_rules_parity(tmp_path):
    exe = build(tmp_path, None)
    kps = _kps_roundtrip(random_kps())
    c = run_exe(exe, kps)
    py_f = features.compute_many(kps)
    np.testing.assert_allclose(c[:, :12], py_f, atol=1e-4)
    py_present = np.array([features.person_present(k) for k in kps])
    np.testing.assert_array_equal(c[:, 12].astype(bool), py_present)
    py_rules = np.array([rules.score(f) for f in py_f])
    np.testing.assert_allclose(c[:, 14], py_rules, atol=1e-6)
    np.testing.assert_allclose(c[:, 13], py_rules, atol=1e-6)  # sans modele : regles


@pytest.mark.parametrize("kind", ["logreg", "mlp"])
def test_model_parity(tmp_path, kind):
    import joblib

    import make_synthetic  # noqa: F401
    import train

    data = tmp_path / "data"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic.py"), "--sessions", "3",
                    "--frames", "300", "--out", str(data)], check=True, capture_output=True)
    model = tmp_path / "clf.joblib"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--data", str(data), "--out", str(model),
                    "--force", kind], check=True, capture_output=True)
    header = tmp_path / "posture_model.h"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "export_c.py"), "--model", str(model), "--out",
                    str(header)], check=True, capture_output=True)
    exe = build(tmp_path, header)
    kps = _kps_roundtrip(random_kps(200, seed=3))
    c = run_exe(exe, kps)
    pipe = joblib.load(model)["pipeline"]
    py_p = pipe.predict_proba(features.compute_many(kps))[:, 1]
    np.testing.assert_allclose(c[:, 13], py_p, atol=1e-4)  # poids exportes en float32
    assert train  # import ok


def test_alert_parity(tmp_path):
    exe = build(tmp_path, None)
    rng = np.random.default_rng(0)
    seq, t = [], 0.0
    for _ in range(600):
        t += rng.uniform(0.5, 2.0)
        present = rng.random() > 0.08
        prob = float(np.clip(0.5 + 0.45 * np.sin(t / 40) + rng.normal(0, 0.1), 0, 1))
        seq.append((t, present, prob))
    inp = "\n".join(f"{t:.6f},{int(p)},{q:.6f}" for t, p, q in seq) + "\n"
    out = subprocess.run([str(exe), "alert"], input=inp, capture_output=True, text=True, check=True).stdout
    c = [line.split(",") for line in out.strip().splitlines()]
    st = AlertState(AlertParams())
    for (t, p, q), (cs, cema, cbeep) in zip(seq, c):
        s, beep = st.update(float(f"{t:.6f}"), float(f"{q:.6f}") if p else None)
        assert s == cs
        assert int(beep) == int(cbeep)
        assert st.ema == pytest.approx(float(cema), abs=1e-6)


# ---------------------------------------------------------------------------
# Etalonnage (calibration.cpp) et rotation d'image (image_rotate.hpp)

from posture import calibration as calib  # noqa: E402


def _fmt(k):
    return ",".join(f"{v:.6f}" for v in k.reshape(-1))


def _calib_case(rng, n_cal=14, n_eval=150, conf01=False):
    """Profil assis a camera fixe, bruit, points manquants ; conf 0/1 comme ESP-DL si conf01."""
    import make_synthetic as ms

    base = ms.sample(rng, "desk")[0].astype(np.float64)
    cal = []
    for _ in range(n_cal):
        k = base.copy()
        k[:, :2] += rng.normal(0, 2.0, size=(17, 2))
        cal.append(k)
    ev = []
    for i in range(n_eval):
        k = base.copy()
        k[:, :2] += rng.normal(0, 25.0, size=(17, 2)) * rng.uniform(0, 1)
        k[:, :2] *= rng.uniform(0.7, 1.4)
        if i % 5 == 0:
            k[rng.integers(0, 7, size=2), :] = 0
        ev.append(k)
    out = [np.asarray(k, np.float32) for k in cal + ev]
    if conf01:
        for k in out:
            k[:, 2] = ((k[:, 0] != 0) | (k[:, 1] != 0)).astype(np.float32)
    out = [_kps_roundtrip(k) for k in out]
    return out[:n_cal], out[n_cal:]


@pytest.mark.parametrize("seed,conf01", [(0, False), (1, False), (2, True), (3, True)])
def test_calibration_parity(tmp_path, seed, conf01):
    exe = build(tmp_path, None)
    rng = np.random.default_rng(seed)
    cal_k, ev_k = _calib_case(rng, conf01=conf01)
    times = [0.25 * i for i in range(len(cal_k))]   # countdown 1 s, duree 2 s -> fin a t >= 3
    lines = [f"C,{t:.3f},{int(features.person_present(k))},{_fmt(k)}" for t, k in zip(times, cal_k)]
    lines += [f"E,{_fmt(k)}" for k in ev_k]
    out = subprocess.run([str(exe), "calib"], input="\n".join(lines) + "\n", capture_output=True,
                         text=True, check=True).stdout.strip().splitlines()

    c = calib.Calibrator(duration_s=2.0, countdown_s=1.0, min_frames=3)
    c.start(times[0])
    b = err = None
    for t, k in zip(times, cal_k):
        present = features.person_present(k)
        b, err = c.update(t, k if present else None, features.compute(k) if present else None)
        if b is not None or err:
            break
    assert b is not None, err
    cb = [float(v) for v in out[0].split(",")[1:]]
    assert cb[0] == 1
    py_b = [*b.ear, *b.sh, b.dirx, b.ref, b.head_nose, b.head_eye, b.n_frames, *b.features]
    np.testing.assert_allclose(cb[1:], py_b, rtol=1e-6, atol=1e-6)

    names = ["sh_drop", "sh_fwd", "d_neck", "head_drop", "head_fwd", "neck_short"]
    n_ok = 0
    for k, line in zip(ev_k, out[1:]):
        v = line.split(",")[1:]
        dev = calib.deviations(b, k, features.compute(k))
        assert int(v[0]) == (dev is not None)
        if dev is None:
            continue
        n_ok += 1
        np.testing.assert_allclose([float(x) for x in v[1:7]], [dev[n] for n in names], rtol=1e-6, atol=1e-6)
        assert float(v[7]) == pytest.approx(dev["scale"], rel=1e-6, abs=1e-6)
        for mode, s_col, w_col in (("trunk", 8, 9), ("all", 10, 11)):
            s = calib.score(dev, mode=mode)
            assert float(v[s_col]) == pytest.approx(-1.0 if s is None else s, rel=1e-6, abs=1e-6)
            w = calib.worst(dev, mode=mode)
            assert names[int(v[w_col])] == w
    assert n_ok > 100


@pytest.mark.parametrize("deg,cv", [(90, "ROTATE_90_CLOCKWISE"), (180, "ROTATE_180"), (270, "ROTATE_90_COUNTERCLOCKWISE")])
def test_rotation_matches_opencv(tmp_path, deg, cv):
    cv2 = pytest.importorskip("cv2")
    exe = build(tmp_path, None)
    w, h = 7, 4
    img = np.arange(w * h, dtype=np.uint16).reshape(h, w)
    inp = f"{w} {h} {deg}\n" + "\n".join(str(int(v)) for v in img.reshape(-1)) + "\n"
    out = subprocess.run([str(exe), "rotate"], input=inp, capture_output=True, text=True, check=True).stdout.split()
    dw, dh = int(out[0]), int(out[1])
    got = np.array([int(v) for v in out[2:]], dtype=np.uint16).reshape(dh, dw)
    np.testing.assert_array_equal(got, cv2.rotate(img, getattr(cv2, cv)))
