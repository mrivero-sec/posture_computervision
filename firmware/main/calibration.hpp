// Etalonnage personnel (miroir EXACT de posture/calibration.py, verifie par
// tests/test_parity.py). C++ pur, sans dependance ESP-IDF : compilable sur PC.
//
// La camera est fixe : on enregistre quelques secondes de bonne posture, puis on
// mesure l'ecart a cette reference en longueurs oreille-epaule. Mode "trunk" :
// les ecarts de la tete ne comptent que si l'epaule a bouge (regarder vers le
// bas ou ecrire ne declenche rien).
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

// Oreille et epaule du cote le mieux vu + direction du regard (0 si inconnue).
SidePoints side_points(const Keypoints &k, double thr = CONF_THR);
// Distances oreille-nez et oreille-oeil (px), 0 si non visibles.
void head_sizes(const Keypoints &k, const Point &ear, double thr, double &dn, double &de);

constexpr uint32_t BASELINE_MAGIC = 0xB0570001;  // a changer si la structure change (NVS)

struct Baseline {
    uint32_t magic = 0;
    Point ear, sh;
    double dirx = 1.0;
    double ref = 0.0;           // distance oreille-epaule de reference (px)
    Features features{};        // mediane des features en bonne posture
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
    // (seuil, echelle) dans l'ordre de DevIdx ; memes valeurs que config.yaml
    double thr[N_DEV] = {0.12, 0.30, 12.0, 0.35, 0.45, 0.15};
    double scl[N_DEV] = {0.07, 0.10, 5.0, 0.15, 0.15, 0.07};
    bool trunk_mode = true;
    double gate_sh_drop = 0.07;
    double gate_sh_fwd = 0.10;
    double scale_min = 0.75;
    double scale_max = 1.33;
};

Deviations deviations(const Baseline &b, const Keypoints &k, double thr = CONF_THR);
// Probabilite de mauvaise posture d'apres l'ecart, ou -1 si non applicable
// (pas de mesure, ou etalonnage perime : echelle hors [scale_min, scale_max]).
double calib_score(const Deviations &d, const CalibParams &p);
// Indice de l'ecart le plus depasse (-1 si aucun).
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
    // k = nullptr si personne absente. Retourne DONE (baseline() valide) ou FAILED a la fin.
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
