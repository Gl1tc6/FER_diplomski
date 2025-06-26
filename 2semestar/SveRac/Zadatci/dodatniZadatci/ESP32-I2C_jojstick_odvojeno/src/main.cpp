#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "AnalogJoystick.h"

static const char* TAG = "JOYSTICK";
AnalogJoystick joystick;

extern "C" void app_main() {
    esp_log_level_set("*", ESP_LOG_INFO);
    
    joystick.init();
    
    ESP_LOGI(TAG, "ESP32 Analog Joystick Reader Started");
    
    while (1) {
        joystick.print_values();
        vTaskDelay(500 / portTICK_PERIOD_MS);
    }
}