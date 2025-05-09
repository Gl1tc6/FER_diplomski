// main.cpp
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "CKeyPad.h"

static const char* TAG = "MAIN";

// Callback funkcije
void onSinglePress(char key) {
    ESP_LOGI(TAG, "Single press: %c", key);
}

void onDoublePress(char key) {
    ESP_LOGI(TAG, "Double press: %c", key);
}

void onLongPress(char key) {
    ESP_LOGI(TAG, "Long press: %c", key);
}

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "Initializing keypad...");
    
    // Instanciranje klase
    CKeyPad keypad;
    
    // Postavljanje callback funkcija
    keypad.setSinglePressCallback(onSinglePress);
    keypad.setDoublePressCallback(onDoublePress);
    keypad.setLongPressCallback(onLongPress);
    
    // Inicijalizacija tipkovnice
    keypad.init();
    
    // Pokretanje skeniranja tipkovnice
    keypad.start();
    
    ESP_LOGI(TAG, "Keypad running. Press keys to test.");
    
    // Glavna petlja
    while (1) {
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
}