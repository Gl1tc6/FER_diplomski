#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "AnalogJoystick.h"

AnalogJoystick joystick;

extern "C" void app_main() {
    joystick.init();
    
    printf("ESP32 Analog Joystick Reader Started\n");
    
    while (1) {
        joystick.print_values();
        vTaskDelay(500 / portTICK_PERIOD_MS);
    }
}