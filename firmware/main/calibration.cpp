// Miroir EXACT de posture/calibration.py (voir calibration.hpp).
#include "calibration.hpp"

#include <algorithm>
#include <cmath>

namespace posture {

const char *const DEV_NAMES[N_DEV] = {"sh_drop", "sh_fwd", "d_neck", "head_drop", "head_fwd", "neck_short"};

static double conf(const Keypoints &k, int i, double thr) { return k[i].c >= thr ? (double)k[i].c : 0.0; }

SidePoints side_points(const Keypoints &k, double thr) {
    SidePoints s;
    double left = conf(k, L_EAR, thr) + conf(k, L_SH, thr);
    double right = conf(k, R_EAR, thr) + conf(k, R_SH, thr);
    int ear_i = left >= right ? L_EAR : R_EAR;
    int sh_i = left >= right ? L_SH : R_SH;
    if (conf(k, ear_i, thr) == 0.0) ear_i = ear_i == L_EAR ? R_EAR : L_EAR;
    if (conf(k, sh_i, thr) == 0.0) sh_i = sh_i == L_SH ? R_SH : L_SH;
    if (conf(k, ear_i, thr) == 0.0 || conf(k, sh_i, thr) == 0.0) return s;
    s.ear = {(double)k[ear_i].x, (double)k[ear_i].y};
    s.sh = {(double)k[sh_i].x, (double)k[sh_i].y};
    const int order[3] = {NOSE, L_EYE, R_EYE};  // nez, sinon un oeil
    for (int i : order) {
        if (conf(k, i, thr) > 0.0 && std::fabs((double)k[i].x - s.ear.x) > 1e-6) {
            s.dirx = (double)k[i].x > s.ear.x ? 1.0 : -1.0;
            break;
        }
    }
    s.ok = true;
    return s;
}

void head_sizes(const Keypoints &k, const Point &ear, double thr, double &dn, double &de) {
    dn = de = 0.0;
    if ((double)k[NOSE].c >= thr) dn = std::hypot((double)k[NOSE].x - ear.x, (double)k[NOSE].y - ear.y);
    int eye = k[R_EYE].c > k[L_EYE].c ? R_EYE : L_EYE;  // comme max() Python : egalite -> gauche
    if ((double)k[eye].c >= thr) de = std::hypot((double)k[eye].x - ear.x, (double)k[eye].y - ear.y);
}

static double neck_angle(const Point &ear, const Point &sh, double dirx) {
    return std::atan2(dirx * (ear.x - sh.x), -(ear.y - sh.y)) * 180.0 / M_PI;
}

Deviations deviations(const Baseline &b, const Keypoints &k, double thr) {
    Deviations d;
    SidePoints sp = side_points(k, thr);
    if (!sp.ok || b.ref <= 1e-6) return d;
    double dn, de;
    head_sizes(k, sp.ear, thr, dn, de);
    double neck = std::hypot(sp.ear.x - sp.sh.x, sp.ear.y - sp.sh.y) / b.ref;
    if (de > 0 && b.head_eye > 0) d.scale = std::sqrt(de / b.head_eye * neck);
    else if (dn > 0 && b.head_nose > 0) d.scale = std::sqrt(dn / b.head_nose * neck);
    else d.scale = 1.0;
    double dirx = b.dirx != 0.0 ? b.dirx : 1.0;
    d.v[D_NECK] = neck_angle(sp.ear, sp.sh, dirx) - neck_angle(b.ear, b.sh, dirx);
    d.v[D_HEAD_DROP] = (sp.ear.y - b.ear.y) / b.ref;
    d.v[D_HEAD_FWD] = dirx * (sp.ear.x - b.ear.x) / b.ref;
    d.v[D_SH_DROP] = (sp.sh.y - b.sh.y) / b.ref;
    d.v[D_SH_FWD] = dirx * (sp.sh.x - b.sh.x) / b.ref;
    d.v[D_NECK_SHORT] = 1.0 - neck;
    d.ok = true;
    return d;
}

static bool head_allowed(const Deviations &d, const CalibParams &p) {
    return !p.trunk_mode || d.v[D_SH_DROP] > p.gate_sh_drop || d.v[D_SH_FWD] > p.gate_sh_fwd;
}

static bool is_head(int i) { return i == D_NECK || i == D_HEAD_DROP || i == D_HEAD_FWD || i == D_NECK_SHORT; }

int calib_worst(const Deviations &d, const CalibParams &p) {
    if (!d.ok) return -1;
    bool head = head_allowed(d, p);
    int best = -1;
    double bv = 0.0;
    for (int i = 0; i < N_DEV; ++i) {
        if (is_head(i) && !head) continue;
        double e = (d.v[i] - p.thr[i]) / p.scl[i];
        if (best < 0 || e > bv) { best = i; bv = e; }
    }
    return best;
}

double calib_score(const Deviations &d, const CalibParams &p) {
    if (!d.ok || d.scale < p.scale_min || d.scale > p.scale_max) return -1.0;
    int w = calib_worst(d, p);
    if (w < 0) return -1.0;
    double e = (d.v[w] - p.thr[w]) / p.scl[w];
    return 1.0 / (1.0 + std::exp(-2.0 * e));
}

// mediane comme numpy (moyenne des deux valeurs centrales si n pair)
static double median(double *v, int n) {
    std::sort(v, v + n);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double median_pos(const double *src, int n) {
    double tmp[Calibrator::MAX_FRAMES];
    int m = 0;
    for (int i = 0; i < n; ++i)
        if (src[i] > 0) tmp[m++] = src[i];
    return m ? median(tmp, m) : 0.0;
}

void Calibrator::start(double now) {
    phase_ = Phase::COUNTDOWN;
    t_start_ = now;
    n_ = 0;
}

Calibrator::Result Calibrator::update(double now, const Keypoints *k, double thr) {
    if (phase_ == Phase::COUNTDOWN && now - t_start_ >= countdown_s_) phase_ = Phase::RECORDING;
    if (phase_ != Phase::RECORDING) return Result::NONE;
    if (k && n_ < MAX_FRAMES) {
        SidePoints sp = side_points(*k, thr);
        if (sp.ok) {
            pts_[n_] = sp;
            feats_[n_] = compute(*k, thr);
            head_sizes(*k, sp.ear, thr, dn_[n_], de_[n_]);
            ++n_;
        }
    }
    if (now - t_start_ < countdown_s_ + duration_s_) return Result::NONE;
    phase_ = Phase::IDLE;
    if (n_ < min_frames_) return Result::FAILED;

    double tmp[MAX_FRAMES];
    Baseline b;
    auto med = [&](auto get) {
        for (int i = 0; i < n_; ++i) tmp[i] = get(i);
        return median(tmp, n_);
    };
    b.ear = {med([&](int i) { return pts_[i].ear.x; }), med([&](int i) { return pts_[i].ear.y; })};
    b.sh = {med([&](int i) { return pts_[i].sh.x; }), med([&](int i) { return pts_[i].sh.y; })};
    int nd = 0;
    for (int i = 0; i < n_; ++i)
        if (pts_[i].dirx != 0.0) tmp[nd++] = pts_[i].dirx;
    b.dirx = (nd == 0 || median(tmp, nd) >= 0) ? 1.0 : -1.0;
    b.ref = med([&](int i) {
        return std::hypot(pts_[i].ear.x - pts_[i].sh.x, pts_[i].ear.y - pts_[i].sh.y);
    });
    for (int j = 0; j < N_FEATURES; ++j) b.features[j] = med([&](int i) { return feats_[i][j]; });
    b.head_nose = median_pos(dn_, n_);
    b.head_eye = median_pos(de_, n_);
    b.n_frames = n_;
    b.magic = BASELINE_MAGIC;
    out_ = b;
    return Result::DONE;
}

}  // namespace posture
