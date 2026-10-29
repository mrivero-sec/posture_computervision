// BAD POSTURE - test camera SANS Wi-Fi, via le cable USB (port COM/UART)
// Le PC envoie 'c' -> la carte capture une image JPEG et renvoie :
//   "FRAME" + longueur (uint32, little-endian) + octets JPEG
// A utiliser avec : python scripts/serial_viewer.py --port COMx
// Ne PAS ouvrir le Moniteur serie d'Arduino en meme temps (port occupe).
//
// Reglages Outils : ESP32S3 Dev Module, PSRAM = OPI PSRAM,
// Flash 16MB, Partition 16M Flash (3MB APP/9.9MB FATFS), USB CDC On Boot = Disabled.

#include "esp_camera.h"

#define BAUD 2000000   // la puce CH343 supporte jusqu'a 6 Mbit/s

// Brochage Freenove ESP32-S3-WROOM CAM (= ESP32-S3-EYE)
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  15
#define SIOD_GPIO_NUM  4
#define SIOC_GPIO_NUM  5
#define Y2_GPIO_NUM 11
#define Y3_GPIO_NUM 9
#define Y4_GPIO_NUM 8
#define Y5_GPIO_NUM 10
#define Y6_GPIO_NUM 12
#define Y7_GPIO_NUM 18
#define Y8_GPIO_NUM 17
#define Y9_GPIO_NUM 16
#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM  7
#define PCLK_GPIO_NUM  13

#define LED_RGB 48

static void rgb(uint8_t r, uint8_t g, uint8_t b) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  rgbLedWrite(LED_RGB, r, g, b);
#else
  neopixelWrite(LED_RGB, r, g, b);
#endif
}

void setup() {
  Serial.setTxBufferSize(16384);
  Serial.begin(BAUD);
  delay(500);

  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer   = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size   = FRAMESIZE_VGA;      // 640x480
  c.jpeg_quality = 12;                 // 10 = meilleur, 63 = pire
  c.fb_count     = 2;
  c.fb_location  = CAMERA_FB_IN_PSRAM;
  c.grab_mode    = CAMERA_GRAB_LATEST;

  if (!psramFound()) {
    Serial.println("ERR PSRAM non detectee : Outils > PSRAM = OPI PSRAM");
    rgb(40, 0, 0);
    while (true) delay(1000);
  }

  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    Serial.printf("ERR esp_camera_init 0x%x (nappe camera bien enfoncee ?)\n", err);
    rgb(40, 0, 0);
    while (true) delay(1000);
  }

  sensor_t *s = esp_camera_sensor_get();
  Serial.printf("OK camera PID=0x%x, VGA JPEG, %d baud\n", s->id.PID, BAUD);
  rgb(0, 30, 0);   // vert = pret
}

void loop() {
  if (!Serial.available()) { delay(1); return; }
  int cmd = Serial.read();
  if (cmd != 'c') return;

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { Serial.println("ERR capture"); return; }

  uint32_t len = fb->len;
  Serial.write((const uint8_t *)"FRAME", 5);
  Serial.write((const uint8_t *)&len, 4);
  Serial.write(fb->buf, fb->len);
  Serial.flush();
  esp_camera_fb_return(fb);
}
