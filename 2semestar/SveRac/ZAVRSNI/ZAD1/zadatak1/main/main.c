#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "sensors.h"

void app_main(void)
{
    init();

    while(1) {
        for(uint8_t i = 100; i <= 200; i++) {
            sensor_send(i);
            vTaskDelay(pdMS_TO_TICKS(1000));
            
            sensor_receive();
            vTaskDelay(pdMS_TO_TICKS(100));
        }
        
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}