// BAD POSTURE: ESP32-S3 firmware (Freenove ESP32-S3-WROOM CAM or Seeed XIAO ESP32S3 Sense)
// camera -> COCOPose (YOLO11n-pose, ESP-DL) -> features -> classifier -> LED / buzzer alert
//
// Button: short press = snooze 15 min; double press = calibrate (good posture
// for a few seconds); long press = switch between desk / practitioner profiles.
// The board's BOOT button also calibrates (handy until the button is wired).
// The calibration is kept in flash (NVS), one per profile.
//
// Serial output (115200):
//   KP,<ms>,x0,y0,...,x16,y16       raw keypoints (for scripts/serial_collect.py)
//   ST,<ms>,<state>,<ema>,<prob>,<neck_incl>,<trunk_incl>,<neck_rel>,<infer_ms>,<source>,<sh_drop>,<sh_fwd>,<scale>,<worst>
//      source = calib (calibration in use) | stale (calibration out of range) | model
// No image ever leaves the board.

#include <cstdio>

#include "calibration.hpp"
#include "classifier.hpp"
#include "config.h"
#include "coco_pose.hpp"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "features.hpp"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "image_rotate.hpp"
#include "led_strip.h"
#include "nvs.h"
#include "nvs_flash.h"

static const char *TAG = "badposture";

using posture::State;

static led_strip_handle_t s_led = nullptr;

// ---------------------------------------------------------------------------
static double now_s() { return esp_timer_get_time() / 1e6; }
static uint32_t now_ms() { return (uint32_t)(esp_timer_get_time() / 1000); }

static esp_err_t camera_init() {
    camera_config_t c = {};
    c.pin_pwdn = CAM_PIN_PWDN;
    c.pin_reset = CAM_PIN_RESET;
    c.pin_xclk = CAM_PIN_XCLK;
    c.pin_sccb_sda = CAM_PIN_SIOD;
    c.pin_sccb_scl = CAM_PIN_SIOC;
    c.pin_d7 = CAM_PIN_D7;
    c.pin_d6 = CAM_PIN_D6;
    c.pin_d5 = CAM_PIN_D5;
    c.pin_d4 = CAM_PIN_D4;
    c.pin_d3 = CAM_PIN_D3;
    c.pin_d2 = CAM_PIN_D2;
    c.pin_d1 = CAM_PIN_D1;
    c.pin_d0 = CAM_PIN_D0;
    c.pin_vsync = CAM_PIN_VSYNC;
    c.pin_href = CAM_PIN_HREF;
    c.pin_pclk = CAM_PIN_PCLK;
    c.xclk_freq_hz = 20000000;
    c.ledc_timer = LEDC_TIMER_0;      // the buzzer uses TIMER_1 / CHANNEL_1
    c.ledc_channel = LEDC_CHANNEL_0;
    c.pixel_format = PIXFORMAT_RGB565;
    c.frame_size = FRAMESIZE_QVGA;    // 320x240: the model letterboxes the input anyway
    c.jpeg_quality = 12;
    c.fb_count = 2;                   // CAMERA_GRAB_LATEST needs at least 2 buffers
    c.fb_location = CAMERA_FB_IN_PSRAM;
    c.grab_mode = CAMERA_GRAB_LATEST; // always the most recent frame
    return esp_camera_init(&c);
}

static void led_init() {
    led_strip_config_t sc = {};
    sc.strip_gpio_num = PIN_LED_WS2812;
    sc.max_leds = 1;
    sc.led_model = LED_MODEL_WS2812;
    led_strip_rmt_config_t rc = {};
    rc.resolution_hz = 10 * 1000 * 1000;
    ESP_ERROR_CHECK(led_strip_new_rmt_device(&sc, &rc, &s_led));
    led_strip_clear(s_led);
}

static void led_rgb(uint8_t r, uint8_t g, uint8_t b) {
    auto k = [](uint8_t v) { return (uint32_t)v * LED_BRIGHTNESS / 255; };
    led_strip_set_pixel(s_led, 0, k(r), k(g), k(b));
    led_strip_refresh(s_led);
}

static void led_show(State s, bool snoozed) {
    if (snoozed && s != State::ABSENT) return led_rgb(0, 0, 255);  // blue = snoozed
    switch (s) {
        case State::ABSENT: led_strip_clear(s_led); break;
        case State::OK: led_rgb(0, 255, 0); break;
        case State::WARNING: led_rgb(255, 120, 0); break;
        case State::ALERT: led_rgb(255, 0, 0); break;
    }
}

static void buzzer_init() {
    ledc_timer_config_t t = {};
    t.speed_mode = LEDC_LOW_SPEED_MODE;
    t.duty_resolution = LEDC_TIMER_10_BIT;
    t.timer_num = LEDC_TIMER_1;
    t.freq_hz = BUZZER_FREQ_HZ;
    t.clk_cfg = LEDC_AUTO_CLK;
    ESP_ERROR_CHECK(ledc_timer_config(&t));
    ledc_channel_config_t ch = {};
    ch.gpio_num = PIN_BUZZER;
    ch.speed_mode = LEDC_LOW_SPEED_MODE;
    ch.channel = LEDC_CHANNEL_1;
    ch.timer_sel = LEDC_TIMER_1;
    ch.duty = 0;
    ESP_ERROR_CHECK(ledc_channel_config(&ch));
}

static void buzz(int ms) {
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1, 512);  // 50 %
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1);
    vTaskDelay(pdMS_TO_TICKS(ms));
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1, 0);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1);
}

static void beep_pattern(int n) {
    for (int i = 0; i < n; ++i) {
        buzz(BUZZER_BEEP_MS);
        vTaskDelay(pdMS_TO_TICKS(120));
    }
}

static void button_init() {
    gpio_config_t io = {};
    io.pin_bit_mask = (1ULL << PIN_BUTTON) | (1ULL << PIN_BUTTON_BOOT);
    io.mode = GPIO_MODE_INPUT;
    io.pull_up_en = GPIO_PULLUP_ENABLE;
    ESP_ERROR_CHECK(gpio_config(&io));
}

// 0 = nothing, 1 = short press, 2 = long press, 3 = double press (detected on release).
// A short press is only reported after DOUBLE_PRESS_MS without a second press.
static int button_poll() {
    static bool was_down = false;
    static uint32_t down_at = 0, short_at = 0;
    static bool pending = false;
    bool down = gpio_get_level((gpio_num_t)PIN_BUTTON) == 0;
    uint32_t t = now_ms();
    int ev = 0;
    if (down && !was_down) down_at = t;
    if (!down && was_down) {
        uint32_t d = t - down_at;
        if (d >= LONG_PRESS_MS) {
            ev = 2;
            pending = false;
        } else if (d >= 40) {  // debounce
            if (pending && t - short_at <= DOUBLE_PRESS_MS) {
                ev = 3;
                pending = false;
            } else {
                pending = true;
                short_at = t;
            }
        }
    }
    if (ev == 0 && pending && !down && t - short_at > DOUBLE_PRESS_MS) {
        ev = 1;
        pending = false;
    }
    was_down = down;
    return ev;
}

// Inference takes several seconds: the button is read in its own task
// (every 10 ms) so no press is missed.
static QueueHandle_t s_btn_queue = nullptr;

static void button_task(void *) {
    bool boot_was_down = false;
    uint32_t boot_down_at = 0;
    while (true) {
        int ev = button_poll();
        if (ev) xQueueSend(s_btn_queue, &ev, 0);
        // board BOOT button (GPIO0): press = calibrate, no wiring needed
        bool boot_down = gpio_get_level((gpio_num_t)PIN_BUTTON_BOOT) == 0;
        if (boot_down && !boot_was_down) boot_down_at = now_ms();
        if (!boot_down && boot_was_down && now_ms() - boot_down_at >= 40) {
            int cal = 3;
            xQueueSend(s_btn_queue, &cal, 0);
        }
        boot_was_down = boot_down;
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ---- Calibration in flash (NVS), one per profile ----------------------------
static void nvs_init() {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
}

static const char *calib_key(int profile) { return profile ? "calib1" : "calib0"; }

static posture::Baseline baseline_load(int profile) {
    posture::Baseline b;
    nvs_handle_t h;
    if (nvs_open("badposture", NVS_READONLY, &h) != ESP_OK) return b;
    size_t len = sizeof(b);
    if (nvs_get_blob(h, calib_key(profile), &b, &len) != ESP_OK || len != sizeof(b) || !b.valid()) b = {};
    nvs_close(h);
    return b;
}

static void baseline_save(int profile, const posture::Baseline &b) {
    nvs_handle_t h;
    if (nvs_open("badposture", NVS_READWRITE, &h) != ESP_OK) return;
    nvs_set_blob(h, calib_key(profile), &b, sizeof(b));
    nvs_commit(h);
    nvs_close(h);
}

static posture::CalibParams calib_params() {
    posture::CalibParams p;  // defaults = config.yaml (trunk mode)
    p.trunk_mode = CALIB_TRUNK_MODE;
    return p;
}

// ---------------------------------------------------------------------------
extern "C" void app_main(void) {
    ESP_LOGI(TAG, "BAD POSTURE starting on %s, classifier = %s", BOARD_NAME,
             POSTURE_HAS_MODEL ? "trained model" : "ISO rules");
    nvs_init();
    ESP_ERROR_CHECK(camera_init());
    led_init();
    buzzer_init();
    button_init();
    s_btn_queue = xQueueCreate(8, sizeof(int));
    xTaskCreate(button_task, "button", 2048, nullptr, 5, nullptr);
    led_rgb(255, 255, 255);

    COCOPose *pose = new COCOPose();  // model selected in menuconfig (V2 QAT by default)
    int profile = 0;                   // 0 = desk, 1 = practitioner
    posture::AlertState alert(profile_desk());
    const posture::CalibParams cparams = calib_params();
    posture::Calibrator calibrator(CALIB_COUNTDOWN_S, CALIB_DURATION_S, CALIB_MIN_FRAMES);
    posture::Baseline baseline = baseline_load(profile);
    ESP_LOGI(TAG, "calibration %s", baseline.valid() ? "loaded from flash" : "missing: double press to calibrate");

    // buffer for the rotated frame (the board stands in portrait)
    uint16_t *rot_buf = nullptr;
    if (CAM_ROTATE_DEG != 0) {
        rot_buf = (uint16_t *)heap_caps_malloc(320 * 240 * 2, MALLOC_CAP_SPIRAM);
        if (!rot_buf) ESP_LOGE(TAG, "not enough memory for rotation: frame not rotated");
    }
    led_strip_clear(s_led);
    beep_pattern(1);

    uint32_t last_frame = 0;
    while (true) {
        int ev = 0;
        xQueueReceive(s_btn_queue, &ev, 0);
        if (ev == 1) {
            alert.snooze(now_s());
            ESP_LOGI(TAG, "snooze 15 min");
            led_show(alert.state(), true);
        } else if (ev == 2) {
            profile = 1 - profile;
            alert.set_params(profile ? profile_practitioner() : profile_desk());
            baseline = baseline_load(profile);
            ESP_LOGI(TAG, "profile = %s, calibration %s", profile ? "practitioner" : "desk",
                     baseline.valid() ? "loaded" : "missing");
            beep_pattern(profile + 1);
        } else if (ev == 3) {
            calibrator.start(now_s());
            ESP_LOGI(TAG, "calibration: sit up straight, recording in %d s", (int)CALIB_COUNTDOWN_S);
            buzz(60);
        }

        // during calibration: white LED (countdown) then cyan (recording)
        if (calibrator.active()) {
            if (calibrator.phase() == posture::Calibrator::Phase::COUNTDOWN) led_rgb(255, 255, 255);
            else led_rgb(0, 200, 255);
        }

        uint32_t period = calibrator.active() ? 0 : FRAME_PERIOD_MS;  // calibration: as fast as possible
        if (now_ms() - last_frame < period) {
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }
        last_frame = now_ms();

        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGW(TAG, "camera capture failed");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        dl::image::img_t img = {};
        img.data = fb->buf;
        img.width = (uint16_t)fb->width;
        img.height = (uint16_t)fb->height;
        img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE;  // esp32-camera outputs big-endian RGB565
        if (rot_buf && fb->width * fb->height <= 320 * 240) {
            int dw, dh;
            posture::rotate_u16((const uint16_t *)fb->buf, fb->width, fb->height, rot_buf, CAM_ROTATE_DEG, dw, dh);
            img.data = rot_buf;
            img.width = (uint16_t)dw;
            img.height = (uint16_t)dh;
        }

        uint32_t t0 = now_ms();
        std::list<dl::detect::result_t> &res = pose->run(img);
        uint32_t infer_ms = now_ms() - t0;
        esp_camera_fb_return(fb);

        // largest person = closest
        const dl::detect::result_t *best = nullptr;
        for (const auto &r : res)
            if (r.keypoint.size() >= 34 && (!best || r.box_area() > best->box_area())) best = &r;

        posture::Keypoints k{};
        if (best) {
            for (int i = 0; i < posture::N_KPS; ++i) {
                k[i].x = (float)best->keypoint[2 * i];
                k[i].y = (float)best->keypoint[2 * i + 1];
                k[i].c = (k[i].x != 0.f || k[i].y != 0.f) ? 1.f : 0.f;  // ESP-DL: (0,0) = not detected
            }
            printf("KP,%lu", (unsigned long)last_frame);
            for (int i = 0; i < posture::N_KPS; ++i) printf(",%d,%d", (int)k[i].x, (int)k[i].y);
            printf("\n");
        }

        bool present = best && posture::person_present(k);
        double t = now_s();

        if (calibrator.active()) {
            auto r = calibrator.update(t, present ? &k : nullptr);
            if (r == posture::Calibrator::Result::DONE) {
                baseline = calibrator.baseline();
                baseline_save(profile, baseline);
                alert.reset();
                ESP_LOGI(TAG, "calibration saved (%d frames)", (int)baseline.n_frames);
                beep_pattern(2);
            } else if (r == posture::Calibrator::Result::FAILED) {
                ESP_LOGW(TAG, "calibration failed: %d usable frames (min %d)", calibrator.n_frames(),
                         (int)CALIB_MIN_FRAMES);
                led_rgb(255, 0, 0);
                beep_pattern(3);
            }
            continue;
        }

        double prob = 0.0, cprob = -1.0;
        posture::Features f{};
        posture::Deviations dev;
        int worst = -1;
        if (present) {
            f = posture::compute(k);
            prob = posture::predict_proba(f);
            if (baseline.valid()) {
                dev = posture::deviations(baseline, k);
                cprob = posture::calib_score(dev, cparams);
                if (cprob >= 0.0) {  // valid calibration: it replaces the classifier
                    prob = cprob;
                    worst = posture::calib_worst(dev, cparams);
                }
            }
        }
        bool beep = alert.update(t, present, prob);
        led_show(alert.state(), alert.snoozed(t));
        if (beep && alert.params().sound) beep_pattern(2);

        printf("ST,%lu,%s,%.3f,%.3f,%.1f,%.1f,%.1f,%lu,%s,%.2f,%.2f,%.2f,%s\n", (unsigned long)last_frame,
               posture::state_name(alert.state()), alert.ema(), prob, f[posture::F_NECK_INCL],
               f[posture::F_TRUNK_INCL], f[posture::F_NECK_REL], (unsigned long)infer_ms,
               cprob >= 0.0 ? "calib" : (baseline.valid() ? "stale" : "model"), dev.v[posture::D_SH_DROP],
               dev.v[posture::D_SH_FWD], dev.scale, worst >= 0 ? posture::DEV_NAMES[worst] : "");
    }
}
