// Harnais de test PC : lit des keypoints (51 floats par ligne, CSV) sur stdin,
// ecrit les 12 features + la probabilite, en utilisant le code C++ du firmware.
// Compile par tests/test_parity.py avec g++.
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

#include <vector>

#include "alert.hpp"
#include "calibration.hpp"
#include "classifier.hpp"
#include "features.hpp"
#include "image_rotate.hpp"

static posture::Keypoints parse_kps(std::stringstream &ss) {
    posture::Keypoints k{};
    std::string cell;
    for (int i = 0; i < 17 * 3 && std::getline(ss, cell, ','); ++i) {
        float v = std::stof(cell);
        if (i % 3 == 0) k[i / 3].x = v;
        else if (i % 3 == 1) k[i / 3].y = v;
        else k[i / 3].c = v;
    }
    return k;
}

int main(int argc, char **argv) {
    std::string mode = argc > 1 ? argv[1] : "features";
    std::string line;
    if (mode == "alert") {
        // entree : t,present,prob ; sortie : etat,ema,beep
        posture::AlertParams p;
        posture::AlertState st(p);
        while (std::getline(std::cin, line)) {
            double t, prob;
            int present;
            if (sscanf(line.c_str(), "%lf,%d,%lf", &t, &present, &prob) != 3) continue;
            bool beep = st.update(t, present != 0, prob);
            printf("%s,%.9f,%d\n", posture::state_name(st.state()), st.ema(), beep ? 1 : 0);
        }
        return 0;
    }
    if (mode == "calib") {
        // C,t,present,<51> = image d'etalonnage ; E,<51> = image a evaluer
        posture::Calibrator cal(1.0, 2.0, 3);
        posture::Baseline b;
        bool started = false;
        while (std::getline(std::cin, line)) {
            std::stringstream ss(line);
            std::string tag, cell;
            std::getline(ss, tag, ',');
            if (tag == "C") {
                std::getline(ss, cell, ',');
                double t = std::stod(cell);
                std::getline(ss, cell, ',');
                int present = std::stoi(cell);
                posture::Keypoints k = parse_kps(ss);
                if (!started) { cal.start(t); started = true; }
                auto r = cal.update(t, present ? &k : nullptr);
                if (r != posture::Calibrator::Result::NONE) {
                    b = cal.baseline();
                    printf("B,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%d", r == posture::Calibrator::Result::DONE ? 1 : 0,
                           b.ear.x, b.ear.y, b.sh.x, b.sh.y, b.dirx, b.ref, b.head_nose, b.head_eye, (int)b.n_frames);
                    for (double v : b.features) printf(",%.9f", v);
                    printf("\n");
                }
            } else if (tag == "E") {
                posture::Keypoints k = parse_kps(ss);
                posture::Deviations d = posture::deviations(b, k);
                posture::CalibParams pt, pa;
                pa.trunk_mode = false;
                printf("D,%d", d.ok ? 1 : 0);
                for (double v : d.v) printf(",%.9f", v);
                printf(",%.9f,%.9f,%d,%.9f,%d\n", d.scale, posture::calib_score(d, pt), posture::calib_worst(d, pt),
                       posture::calib_score(d, pa), posture::calib_worst(d, pa));
            }
        }
        return 0;
    }
    if (mode == "rotate") {
        int w, h, deg;
        if (!(std::cin >> w >> h >> deg)) return 1;
        std::vector<uint16_t> src(w * h), dst(w * h);
        for (auto &v : src) { int x; std::cin >> x; v = (uint16_t)x; }
        int dw, dh;
        posture::rotate_u16(src.data(), w, h, dst.data(), deg, dw, dh);
        printf("%d %d\n", dw, dh);
        for (auto v : dst) printf("%d\n", v);
        return 0;
    }
    while (std::getline(std::cin, line)) {
        std::stringstream ss(line);
        posture::Keypoints k{};
        std::string cell;
        for (int i = 0; i < 17 * 3 && std::getline(ss, cell, ','); ++i) {
            float v = std::stof(cell);
            if (i % 3 == 0) k[i / 3].x = v;
            else if (i % 3 == 1) k[i / 3].y = v;
            else k[i / 3].c = v;
        }
        posture::Features f = posture::compute(k);
        for (double v : f) printf("%.9f,", v);
        printf("%d,%.9f,%.9f\n", posture::person_present(k) ? 1 : 0, posture::predict_proba(f), posture::rules_score(f));
    }
    return 0;
}
