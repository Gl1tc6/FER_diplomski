#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "ckeypad.h"

static const char TAG[] = "MAIN";

// Global keypad instance
static CKeyPad keypad;

// Callback functions for different press types
void on_single_press(char key, press_type_t press_type) {
    printf("SINGLE PRESS: Key '%c' was pressed once\n", key);
    ESP_LOGI(TAG, "SINGLE PRESS: Key '%c'", key);
}

void on_double_press(char key, press_type_t press_type) {
    printf("DOUBLE PRESS: Key '%c' was pressed twice\n", key);
    ESP_LOGI(TAG, "DOUBLE PRESS: Key '%c'", key);
}

void on_long_press(char key, press_type_t press_type) {
    printf("LONG PRESS: Key '%c' was held down\n", key);
    ESP_LOGI(TAG, "LONG PRESS: Key '%c'", key);
}

void app_main(void) {
    ESP_LOGI(TAG, "Starting CKeyPad Demo");
    
    // Initialize the keypad
    ckeypad_init(&keypad);
    
    // Set up callback functions
    ckeypad_set_single_press_callback(&keypad, on_single_press);
    ckeypad_set_double_press_callback(&keypad, on_double_press);
    ckeypad_set_long_press_callback(&keypad, on_long_press);
    
    // Start keypad scanning
    ckeypad_start(&keypad);
    
    ESP_LOGI(TAG, "CKeyPad Demo started. Press keys on the keypad to test:");
    ESP_LOGI(TAG, "- Single press: Quick press and release");
    ESP_LOGI(TAG, "- Double press: Press twice quickly (within 300ms)");
    ESP_LOGI(TAG, "- Long press: Hold key for more than 500ms");
    
    // Main loop - just keep the program running
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        // You could add other main program logic here
    }
}