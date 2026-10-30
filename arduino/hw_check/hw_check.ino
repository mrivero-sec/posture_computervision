// BAD POSTURE - hardware check for the Freenove ESP32-S3-WROOM CAM board
// Expected for an N16R8: 16 MB flash, ~8 MB PSRAM.
// Blue LED on GPIO2 + WS2812 RGB LED on GPIO48 cycling red/green/blue.

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
  Serial.printf("Chip      : %s rev %d, %d cores, %lu MHz\n",
                ESP.getChipModel(), ESP.getChipRevision(),
                ESP.getChipCores(), (unsigned long)ESP.getCpuFreqMHz());
  Serial.printf("Flash     : %lu MB\n", (unsigned long)(ESP.getFlashChipSize() / (1024 * 1024)));
  if (psramFound()) {
    Serial.printf("PSRAM     : %lu KB (free %lu KB)\n",
                  (unsigned long)(ESP.getPsramSize() / 1024),
                  (unsigned long)(ESP.getFreePsram() / 1024));
  } else {
    Serial.println("PSRAM     : NOT DETECTED -> check Tools > PSRAM = OPI PSRAM");
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
