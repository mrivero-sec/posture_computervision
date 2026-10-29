// BAD POSTURE - verification materielle de la carte Freenove ESP32-S3-WROOM CAM
// Attendu pour une N16R8 : flash 16 Mo, PSRAM ~8 Mo.
// LED bleue GPIO2 + LED RGB WS2812 GPIO48 qui cycle rouge/vert/bleu.

#define LED_BLUE 2
#define LED_RGB  48

static void rgb(uint8_t r, uint8_t g, uint8_t b) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  rgbLedWrite(LED_RGB, r, g, b);
#else
  neopixelWrite(LED_RGB, r, g, b);
#endif
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  pinMode(LED_BLUE, OUTPUT);

  Serial.println();
  Serial.println("=== BAD POSTURE hw_check ===");
  Serial.printf("Puce      : %s rev %d, %d coeurs, %lu MHz\n",
                ESP.getChipModel(), ESP.getChipRevision(),
                ESP.getChipCores(), (unsigned long)ESP.getCpuFreqMHz());
  Serial.printf("Flash     : %lu Mo\n", (unsigned long)(ESP.getFlashChipSize() / (1024 * 1024)));
  if (psramFound()) {
    Serial.printf("PSRAM     : %lu Ko (libre %lu Ko)\n",
                  (unsigned long)(ESP.getPsramSize() / 1024),
                  (unsigned long)(ESP.getFreePsram() / 1024));
  } else {
    Serial.println("PSRAM     : NON DETECTEE -> verifier Outils > PSRAM = OPI PSRAM");
  }
  Serial.printf("Arduino-ESP32 : %d.%d.%d\n", ESP_ARDUINO_VERSION_MAJOR,
                ESP_ARDUINO_VERSION_MINOR, ESP_ARDUINO_VERSION_PATCH);
}

void loop() {
  static uint8_t step = 0;
  digitalWrite(LED_BLUE, step % 2);
  switch (step % 3) {
    case 0: rgb(40, 0, 0); break;
    case 1: rgb(0, 40, 0); break;
    case 2: rgb(0, 0, 40); break;
  }
  step++;
  delay(700);
}
