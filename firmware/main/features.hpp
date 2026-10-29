// Features posturales depuis 17 keypoints COCO.
// Miroir EXACT de posture/features.py (verifie par tests/test_parity.py).
// C++ pur, sans dependance ESP-IDF : compilable sur PC pour les tests.
#pragma once

#include <array>

namespace posture {

constexpr int N_KPS = 17;
constexpr int N_FEATURES = 12;
constexpr double CONF_THR = 0.5;

enum Kp { NOSE = 0, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_ELB, R_ELB, L_WR, R_WR, L_HIP, R_HIP };

// x, y en pixels ; c = confiance (sur ESP-DL : 1 si le point est non nul, sinon 0)
struct Keypoint {
    float x = 0.f, y = 0.f, c = 0.f;
};
using Keypoints = std::array<Keypoint, N_KPS>;
using Features = std::array<double, N_FEATURES>;

enum FeatureIdx {
    F_NECK_INCL = 0, F_TRUNK_INCL, F_NECK_REL, F_HEAD_FWD, F_NOSE_DROP, F_SHOULDER_TILT,
    F_HEAD_TILT, F_LATERAL_LEAN, F_EAR_SH_GAP, F_NOSE_OFFSET, F_VIEW_RATIO, F_HIPS_VISIBLE
};

bool person_present(const Keypoints &k, double thr = CONF_THR);
Features compute(const Keypoints &k, double thr = CONF_THR);

}  // namespace posture
