// Hardware configuration + profiles (same values as config.yaml).
// Board selected in menuconfig: "BAD POSTURE > Target board" (Freenove by default).
#pragma once

#include "alert.hpp"
#include "sdkconfig.h"

#if defined(CONFIG_BADPOSTURE_BOARD_XIAO)
// ---- Seeed XIAO ESP32S3 Sense (official Seeed pinout) -----------------------
#define BOARD_NAME     "XIAO ESP32S3 Sense"
#define CAM_PIN_PWDN   -1
#define CAM_PIN_RESET  -1
#define CAM_PIN_XCLK   10
#define CAM_PIN_SIOD   40
#define CAM_PIN_SIOC   39
#define CAM_PIN_D7     48
#define CAM_PIN_D6     11
#define CAM_PIN_D5     12
#define CAM_PIN_D4     14
#define CAM_PIN_D3     16
#define CAM_PIN_D2     18
#define CAM_PIN_D1     17
#define CAM_PIN_D0     15
#define CAM_PIN_VSYNC  38
#define CAM_PIN_HREF   47
#define CAM_PIN_PCLK   13
// Alerts on D0, D1, D2 (external WS2812 LED)
#define PIN_LED_WS2812  1
#define PIN_BUZZER      2
#define PIN_BUTTON      3

#else
// ---- Freenove ESP32-S3-WROOM CAM (N16R8) ------------------------------------
// Camera pinout identical to the ESP32-S3-EYE (Freenove / CameraWebServer examples)
#define BOARD_NAME     "Freenove ESP32-S3-WROOM CAM"
#define CAM_PIN_PWDN   -1
#define CAM_PIN_RESET  -1
#define CAM_PIN_XCLK   15
#define CAM_PIN_SIOD   4
#define CAM_PIN_SIOC   5
#define CAM_PIN_D7     16
#define CAM_PIN_D6     17
#define CAM_PIN_D5     18
#define CAM_PIN_D4     12
#define CAM_PIN_D3     10
#define CAM_PIN_D2     8
#define CAM_PIN_D1     9
#define CAM_PIN_D0     11
#define CAM_PIN_VSYNC  6
#define CAM_PIN_HREF   7
#define CAM_PIN_PCLK   13
// WS2812 RGB LED soldered on the board. Buzzer and button on the breadboard.
// Pins avoided: camera (4-13, 15-18), USB (19, 20), PSRAM (35-37),
// SD card (38-40), UART0 (43, 44), strapping (0, 3, 45, 46), LED (48).
#define PIN_LED_WS2812  48
#define PIN_BUZZER      21
#define PIN_BUTTON      14
#endif

#define LED_BRIGHTNESS  40  // 0-255, the LED is in your field of view: keep it soft

#define BUZZER_FREQ_HZ  2000
#define BUZZER_BEEP_MS  200

// Pace: at most one inference every FRAME_PERIOD_MS
#define FRAME_PERIOD_MS 1000
#define LONG_PRESS_MS   2000   // long press = switch profile; short = snooze 15 min
#define DOUBLE_PRESS_MS 400    // double press = calibrate
#define PIN_BUTTON_BOOT 0      // BOOT button soldered on the board (Freenove and XIAO): calibrate

// ---- Calibration (same rules as posture/calibration.py, trunk mode) ----------
// On the ESP32-S3, coco_pose (640x640) takes ~4.4 s per frame (Espressif docs):
// 20 s of recording gives about 4 frames (vs 5 s and ~25 frames on the PC).
#define CALIB_COUNTDOWN_S 3.0
#define CALIB_DURATION_S  20.0
#define CALIB_MIN_FRAMES  3
#define CALIB_TRUNK_MODE  true   // the head only counts if the shoulder has moved

// ---- Image rotation (menuconfig: BAD POSTURE > Camera rotation) -------------
#if defined(CONFIG_BADPOSTURE_ROTATE_90)
#define CAM_ROTATE_DEG 90
#elif defined(CONFIG_BADPOSTURE_ROTATE_180)
#define CAM_ROTATE_DEG 180
#elif defined(CONFIG_BADPOSTURE_ROTATE_270)
#define CAM_ROTATE_DEG 270
#else
#define CAM_ROTATE_DEG 0
#endif

// ---- Profiles ---------------------------------------------------------------
inline posture::AlertParams profile_desk() {
    posture::AlertParams p;
    p.alpha = 0.3; p.enter_thr = 0.70; p.exit_thr = 0.40; p.trigger_s = 30;
    p.exit_hold_s = 5; p.beep_interval_s = 60; p.sound = true;
    return p;
}

inline posture::AlertParams profile_practitioner() {
    posture::AlertParams p;
    p.alpha = 0.2; p.enter_thr = 0.75; p.exit_thr = 0.45; p.trigger_s = 120;
    p.exit_hold_s = 5; p.beep_interval_s = 120; p.sound = false;  // LED only: a patient is present
    return p;
}
