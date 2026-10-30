// Personal calibration (EXACT mirror of posture/calibration.py, checked by
// tests/test_parity.py). Plain C++, no ESP-IDF dependency: builds on a PC.
//
// The camera is fixed: a few seconds of good posture are recorded, then the
// deviation from that reference is measured in ear-shoulder lengths. "trunk" mode:
// head deviations only count if the shoulder has moved (looking down or
// writing triggers nothing).
#pragma once

#include <cstdint>

#include "features.hpp"

namespace posture {

struct Point {
    double x = 0.0, y = 0.0;
};

struct SidePoints {
    Point ear, sh;
    double dirx = 0.0;
    bool ok = false;
};

// Ear and shoulder on the best-seen side + facing direction (0 if unknown).
SidePoints side_points(const Keypoints &k, double thr = CONF_THR);
// Ear-nose and ear-eye distances (px), 0 if not visible.
void head_sizes(const Keypoints &k, const Point &ear, double thr, double &dn, double &de);

constexpr uint32_t BASELINE_MAGIC = 0xB0570001;  // change it if the struct changes (NVS)

struct Baseline {
    uint32_t magic = 0;
    Point ear, sh;
    double dirx = 1.0;
    double ref = 0.0;           // reference ear-shoulder distance (px)
    Features features{};        // median features in good posture
    double head_nose = 0.0;
    double head_eye = 0.0;
    int32_t n_frames = 0;
    bool valid() const { return magic == BASELINE_MAGIC && ref > 1e-6; }
};

enum DevIdx { D_SH_DROP = 0, D_SH_FWD, D_NECK, D_HEAD_DROP, D_HEAD_FWD, D_NECK_SHORT, N_DEV };
extern const char *const DEV_NAMES[N_DEV];

struct Deviations {
    double v[N_DEV] = {};
    double scale = 1.0;
    bool ok = false;
};

struct CalibParams {
    // (threshold, scale) in DevIdx order; same values as config.yaml
    double thr[N_DEV] = {0.12, 0.30, 12.0, 0.35, 0.45, 0.15};
    double scl[N_DEV] = {0.07, 0.10, 5.0, 0.15, 0.15, 0.07};
    bool trunk_mode = true;
    double gate_sh_drop = 0.07;
    double gate_sh_fwd = 0.10;
    double scale_min = 0.75;
    double scale_max = 1.33;
};

Deviations deviations(const Baseline &b, const Keypoints &k, double thr = CONF_THR);
// Bad posture probability from the deviations, or -1 if not applicable
// (no measurement, or stale calibration: scale outside [scale_min, scale_max]).
double calib_score(const Deviations &d, const CalibParams &p);
// Index of the most exceeded deviation (-1 if none).
int calib_worst(const Deviations &d, const CalibParams &p);

class Calibrator {
public:
    enum class Phase { IDLE, COUNTDOWN, RECORDING };
    enum class Result { NONE, DONE, FAILED };
    static constexpr int MAX_FRAMES = 128;

    Calibrator(double countdown_s = 3.0, double duration_s = 5.0, int min_frames = 8)
        : countdown_s_(countdown_s), duration_s_(duration_s), min_frames_(min_frames) {}
    void start(double now);
    bool active() const { return phase_ != Phase::IDLE; }
    Phase phase() const { return phase_; }
    int n_frames() const { return n_; }
    // k = nullptr if nobody is in view. Returns DONE (baseline() valid) or FAILED at the end.
    Result update(double now, const Keypoints *k, double thr = CONF_THR);
    const Baseline &baseline() const { return out_; }

private:
    double countdown_s_, duration_s_;
    int min_frames_;
    Phase phase_ = Phase::IDLE;
    double t_start_ = 0.0;
    int n_ = 0;
    SidePoints pts_[MAX_FRAMES];
    Features feats_[MAX_FRAMES];
    double dn_[MAX_FRAMES], de_[MAX_FRAMES];
    Baseline out_;
};

}  // namespace posture
