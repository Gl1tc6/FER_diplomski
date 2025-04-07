#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define LED_GPIO 2  // Change to your board's LED pin

void app_main() {
    // Initialize GPIO
    gpio_reset_pin(LED_GPIO);
    gpio_set_direction(LED_GPIO, GPIO_MODE_OUTPUT);
    
    while(1) {
        // Blink the LED (now using pdMS_TO_TICKS)
        gpio_set_level(LED_GPIO, 1);
        vTaskDelay(pdMS_TO_TICKS(500));  // 500ms delay
        gpio_set_level(LED_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(500));  // 500ms delay
        
        ESP_LOGI("LED", "Blinking LED");
    }
}