idf_component_register(
    SRCS "main.cpp" "features.cpp" "alert.cpp" "calibration.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_gpio esp_driver_ledc esp_timer nvs_flash
)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-missing-field-initializers)
