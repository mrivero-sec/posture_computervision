// Probabilite de mauvaise posture : classifieur exporte (posture_model.h)
// ou, s'il n'existe pas encore, baseline par regles ISO 11226 / EN 1005-4.
// Miroir de posture/model.py + posture/rules.py.
#pragma once

#include <algorithm>
#include <cmath>

#include "features.hpp"

#if defined(POSTURE_FORCE_RULES)
#define POSTURE_HAS_MODEL 0
#elif defined(POSTURE_MODEL_HEADER)
#include POSTURE_MODEL_HEADER
#define POSTURE_HAS_MODEL 1
#elif __has_include("posture_model.h")
#include "posture_model.h"
#define POSTURE_HAS_MODEL 1
#else
#define POSTURE_HAS_MODEL 0
#endif

namespace posture {

inline double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

// --- Regles : score = 0.5 pile au seuil ------------------------------------
inline double rules_score(const Features &f) {
    const bool side_view = f[F_VIEW_RATIO] < 0.5;
    const bool hips = f[F_HIPS_VISIBLE] >= 0.5;
    double s = -1e9;
    bool any = false;
    auto add = [&](double v, double thr, double scale) { s = std::max(s, (v - thr) / scale); any = true; };
    add(f[F_NECK_INCL], 25.0, 10.0);
    add(f[F_NECK_REL], 25.0, 10.0);
    if (hips) add(f[F_TRUNK_INCL], 20.0, 10.0);
    if (!side_view) {
        add(f[F_HEAD_TILT], 10.0, 5.0);
        add(f[F_SHOULDER_TILT], 10.0, 5.0);
    }
    return any ? sigmoid(2.0 * s) : 0.0;
}

// --- Classifieur exporte ----------------------------------------------------
inline double predict_proba(const Features &f) {
#if POSTURE_HAS_MODEL
    static_assert(POSTURE_N_FEATURES == N_FEATURES, "posture_model.h ne correspond pas aux features");
    double x[N_FEATURES];
    for (int i = 0; i < N_FEATURES; ++i) x[i] = (f[i] - POSTURE_MEAN[i]) / POSTURE_SCALE[i];
#if POSTURE_MODEL_IS_MLP
    double z = POSTURE_B2;
    for (int j = 0; j < POSTURE_HIDDEN; ++j) {
        double h = POSTURE_B1[j];
        for (int i = 0; i < N_FEATURES; ++i) h += x[i] * POSTURE_W1[i * POSTURE_HIDDEN + j];
        if (h > 0.0) z += h * POSTURE_W2[j];  // ReLU
    }
#else
    double z = POSTURE_B;
    for (int i = 0; i < N_FEATURES; ++i) z += x[i] * POSTURE_W[i];
#endif
    return sigmoid(z);
#else
    return rules_score(f);
#endif
}

}  // namespace posture
