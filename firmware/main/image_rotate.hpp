// Rotates a 16-bit/pixel (RGB565) image in 90 deg steps, clockwise.
// Same conventions as cv2.rotate (90 = ROTATE_90_CLOCKWISE,
// 270 = ROTATE_90_COUNTERCLOCKWISE), checked by tests/test_parity.py.
// With the board in portrait, the person must appear upright, as on the PC.
#pragma once

#include <cstdint>

namespace posture {

// dst must hold w*h pixels. Returns the output size in dw, dh.
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
