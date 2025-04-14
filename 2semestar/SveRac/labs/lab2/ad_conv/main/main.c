#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "sensors.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define B_PIN GPIO_NUM_13 

void app_main(void)
{
    init();
    gpio_set_pull_mode(B_PIN, GPIO_PULLUP_ONLY);

    while (1)
    {
        if(gpio_get_level(B_PIN) == 0){
            float temp = adc_temp();
            ESP_LOGI("VMA320", "Temperature:\t%.1f °C       ", temp);
            temp = dht_temp();
            ESP_LOGI("DHT22 ", "Temperature:\t%.1f °C       ", temp);
            
            vTaskDelay(pdMS_TO_TICKS(150)); //debounce
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    
}
