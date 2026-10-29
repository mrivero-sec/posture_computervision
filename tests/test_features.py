import math

import numpy as np
import pytest

from posture import features, rules
from posture.keypoints import L_EAR, L_EYE, L_HIP, L_SH, NOSE, R_EAR, R_EYE, R_HIP, R_SH, empty_kps

IDX = {n: i for i, n in enumerate(features.FEATURE_NAMES)}


def side_view(trunk_deg, neck_deg, facing=1.0, hips=True):
    """Profil gauche : hanche (300,400), tronc 200 px, cou 80 px."""
    k = empty_kps()
    hx, hy = 300.0, 400.0
    sx = hx + facing * 200 * math.sin(math.radians(trunk_deg))
    sy = hy - 200 * math.cos(math.radians(trunk_deg))
    a = trunk_deg + neck_deg
    ex = sx + facing * 80 * math.sin(math.radians(a))
    ey = sy - 80 * math.cos(math.radians(a))
    k[L_SH] = (sx, sy, 0.9)
    k[L_EAR] = (ex, ey, 0.9)
    k[NOSE] = (ex + facing * 35, ey + 8, 0.9)
    k[L_EYE] = (ex + facing * 25, ey - 5, 0.9)
    if hips:
        k[L_HIP] = (hx, hy, 0.9)
    return k


@pytest.mark.parametrize("facing", [1.0, -1.0])
def test_side_angles(facing):
    f = features.compute(side_view(15, 30, facing))
    assert f[IDX["trunk_incl"]] == pytest.approx(15, abs=1e-3)
    assert f[IDX["neck_incl"]] == pytest.approx(45, abs=1e-3)
    assert f[IDX["neck_rel"]] == pytest.approx(30, abs=1e-3)
    assert f[IDX["head_fwd"]] > 0
    assert f[IDX["view_ratio"]] == 0.0
    assert f[IDX["hips_visible"]] == 1.0


def test_scale_invariance():
    k = side_view(10, 20)
    k2 = k.copy()
    k2[:, :2] *= 2.0
    np.testing.assert_allclose(features.compute(k), features.compute(k2), atol=1e-9)


def test_hips_hidden_fallback():
    f = features.compute(side_view(10, 20, hips=False))
    assert f[IDX["hips_visible"]] == 0.0
    assert f[IDX["trunk_incl"]] == 0.0
    assert f[IDX["neck_rel"]] == pytest.approx(30, abs=1e-3)


def test_front_tilts():
    k = empty_kps()
    k[L_SH], k[R_SH] = (400, 310, 0.9), (200, 300, 0.9)
    k[L_HIP], k[R_HIP] = (360, 500, 0.9), (240, 500, 0.9)
    k[L_EYE], k[R_EYE] = (320, 160, 0.9), (280, 150, 0.9)
    k[L_EAR], k[R_EAR] = (350, 170, 0.9), (250, 160, 0.9)
    k[NOSE] = (300, 180, 0.9)
    f = features.compute(k)
    assert f[IDX["shoulder_tilt"]] == pytest.approx(math.degrees(math.atan2(10, 200)), abs=1e-3)
    assert f[IDX["head_tilt"]] == pytest.approx(math.degrees(math.atan2(10, 40)), abs=1e-3)
    assert f[IDX["view_ratio"]] > 0.5


def test_absent():
    assert not features.person_present(empty_kps())
    assert features.compute(empty_kps()).shape == (features.N_FEATURES,)


def test_rules_order():
    good = rules.score(features.compute(side_view(5, 10)))
    bad = rules.score(features.compute(side_view(30, 40)))
    assert good < 0.5 < bad
