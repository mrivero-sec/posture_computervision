// Rotation d'une image 16 bits/pixel (RGB565) par pas de 90 deg, sens horaire.
// Memes conventions que cv2.rotate (90 = ROTATE_90_CLOCKWISE,
// 270 = ROTATE_90_COUNTERCLOCKWISE), verifie par tests/test_parity.py.
// La carte posee en portrait doit voir la personne debout, comme sur PC.
#pragma once

#include <cstdint>

namespace posture {

// dst doit contenir w*h pixels. Retourne les dimensions de sortie dans dw, dh.
inline void rotate_u16(const uint16_t *src, int w, int h, uint16_t *dst, int deg, int &dw, int &dh) {
    if (deg == 90 || deg == 270) { dw = h; dh = w; }
    else { dw = w; dh = h; }
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            int sx, sy;
            switch (deg) {
                case 90: sx = y; sy = h - 1 - x; break;
                case 180: sx = w - 1 - x; sy = h - 1 - y; break;
                case 270: sx = w - 1 - y; sy = x; break;
                default: sx = x; sy = y; break;
            }
            dst[y * dw + x] = src[sy * w + sx];
        }
    }
}

}  // namespace posture
