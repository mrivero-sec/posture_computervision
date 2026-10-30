#include "features.hpp"

#include <cmath>

namespace posture {
namespace {

constexpr double EPS = 1e-6;
constexpr double PI = 3.14159265358979323846;

struct P {
    double x = 0.0, y = 0.0;
    bool ok = false;
};

inline double clip(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline double deg(double r) { return r * 180.0 / PI; }

P pt(const Keypoints &k, int i, double thr) {
    P p;
    p.x = k[i].x;
    p.y = k[i].y;
    p.ok = (double)k[i].c >= thr && !(p.x == 0.0 && p.y == 0.0);
    return p;
}

P mid(const P &a, const P &b) {
    P m;
    if (a.ok && b.ok) {
        m.x = (a.x + b.x) / 2.0;
        m.y = (a.y + b.y) / 2.0;
        m.ok = true;
    } else if (a.ok) {
        m = a;
    } else if (b.ok) {
        m = b;
    }
    return m;
}

inline double dist(const P &a, const P &b) { return std::hypot(a.x - b.x, a.y - b.y); }

}  // namespace

bool person_present(const Keypoints &k, double thr) {
    bool head = pt(k, NOSE, thr).ok || pt(k, L_EAR, thr).ok || pt(k, R_EAR, thr).ok;
    bool sh = pt(k, L_SH, thr).ok || pt(k, R_SH, thr).ok;
    return head && sh;
}

Features compute(const Keypoints &k, double thr) {
    P nose = pt(k, NOSE, thr);
    P leye = pt(k, L_EYE, thr), reye = pt(k, R_EYE, thr);
    P lear = pt(k, L_EAR, thr), rear = pt(k, R_EAR, thr);
    P lsh = pt(k, L_SH, thr), rsh = pt(k, R_SH, thr);
    P lhip = pt(k, L_HIP, thr), rhip = pt(k, R_HIP, thr);

    int score_l = (int)lear.ok + (int)lsh.ok + (int)lhip.ok;
    int score_r = (int)rear.ok + (int)rsh.ok + (int)rhip.ok;
    P ear, sh, hip, ear2, sh2, hip2;
    if (score_l >= score_r) {
        ear = lear; sh = lsh; hip = lhip; ear2 = rear; sh2 = rsh; hip2 = rhip;
    } else {
        ear = rear; sh = rsh; hip = rhip; ear2 = lear; sh2 = lsh; hip2 = lhip;
    }
    if (!ear.ok) ear = ear2;
    if (!sh.ok) sh = sh2;
    if (!hip.ok) hip = hip2;

    P mid_sh = mid(lsh, rsh), mid_hip = mid(lhip, rhip), mid_ear = mid(lear, rear);
    double hips_visible = mid_hip.ok ? 1.0 : 0.0;

    double dirx = 1.0;
    if (nose.ok && ear.ok && std::fabs(nose.x - ear.x) > EPS) {
        dirx = nose.x > ear.x ? 1.0 : -1.0;
    } else if (nose.ok && sh.ok && std::fabs(nose.x - sh.x) > EPS) {
        dirx = nose.x > sh.x ? 1.0 : -1.0;
    } else {
        // nose hidden: use the eyes, otherwise assume the head is ahead of the shoulder
        P eye = mid(leye, reye);
        if (eye.ok && ear.ok && std::fabs(eye.x - ear.x) > EPS) {
            dirx = eye.x > ear.x ? 1.0 : -1.0;
        } else if (ear.ok && sh.ok && std::fabs(ear.x - sh.x) > EPS) {
            dirx = ear.x > sh.x ? 1.0 : -1.0;
        }
    }

    double sh_w = (lsh.ok && rsh.ok) ? dist(lsh, rsh) : 0.0;
    double L;
    if (mid_hip.ok && mid_sh.ok) L = dist(mid_sh, mid_hip);
    else if (ear.ok && sh.ok) L = 2.5 * dist(ear, sh);
    else if (sh_w > 0.0) L = sh_w;
    else L = 1.0;
    if (L < EPS) L = 1.0;

    double neck_incl = 0.0, head_fwd = 0.0;
    if (ear.ok && sh.ok) {
        double vx = ear.x - sh.x, vy = ear.y - sh.y;
        neck_incl = deg(std::atan2(dirx * vx, -vy));
        head_fwd = dirx * vx / L;
    }
    double trunk_incl = 0.0;
    if (sh.ok && hip.ok) {
        double vx = sh.x - hip.x, vy = sh.y - hip.y;
        trunk_incl = deg(std::atan2(dirx * vx, -vy));
    }
    double neck_rel = hips_visible > 0.0 ? neck_incl - trunk_incl : neck_incl;
    double nose_drop = (nose.ok && ear.ok) ? (nose.y - ear.y) / L : 0.0;

    double shoulder_tilt = 0.0;
    if (lsh.ok && rsh.ok) shoulder_tilt = std::fabs(deg(std::atan2(lsh.y - rsh.y, std::fabs(lsh.x - rsh.x) + EPS)));
    double head_tilt = 0.0;
    if (leye.ok && reye.ok) head_tilt = std::fabs(deg(std::atan2(leye.y - reye.y, std::fabs(leye.x - reye.x) + EPS)));
    else if (lear.ok && rear.ok) head_tilt = std::fabs(deg(std::atan2(lear.y - rear.y, std::fabs(lear.x - rear.x) + EPS)));
    double lateral_lean = 0.0;
    if (mid_sh.ok && mid_hip.ok) lateral_lean = std::fabs(deg(std::atan2(mid_sh.x - mid_hip.x, -(mid_sh.y - mid_hip.y))));

    double ear_sh_gap = (mid_sh.ok && mid_ear.ok) ? (mid_sh.y - mid_ear.y) / L : 0.0;
    double nose_offset = (nose.ok && mid_sh.ok) ? std::fabs(nose.x - mid_sh.x) / L : 0.0;
    double view_ratio = sh_w / L;

    Features f = {
        clip(neck_incl, -90.0, 120.0),
        clip(trunk_incl, -90.0, 120.0),
        clip(neck_rel, -120.0, 150.0),
        clip(head_fwd, -2.0, 2.0),
        clip(nose_drop, -2.0, 2.0),
        clip(shoulder_tilt, 0.0, 90.0),
        clip(head_tilt, 0.0, 90.0),
        clip(lateral_lean, 0.0, 120.0),
        clip(ear_sh_gap, -2.0, 2.0),
        clip(nose_offset, 0.0, 3.0),
        clip(view_ratio, 0.0, 3.0),
        hips_visible,
    };
    return f;
}

}  // namespace posture
