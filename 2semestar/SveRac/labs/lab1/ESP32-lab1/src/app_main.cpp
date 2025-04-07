/* Blink Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "sdkconfig.h"
#include "CLed.h"
#include "CButton.h"

static const char *TAG = "MAIN";

// GPIO
#define BLINK_GPIO 2      // LED pin
#define BUTTON_GPIO 14     // Button pin

// pomocna var
const char* getLedStateName(LedStatus state) {
    switch(state) {
        case LedStatus::OFF: return "OFF";
        case LedStatus::ON: return "ON";
        case LedStatus::BLINK: return "BLINK";
        case LedStatus::FAST_BLINK: return "FAST_BLINK";
        case LedStatus::SLOW_BLINK: return "SLOW_BLINK";
        default: return "UNKNOWN";
    }
}

// LED pointer za callback
CLed* ledPtr = NULL;
LedStatus currentLedState = LedStatus::OFF;

// Print current state
void printCurrentState() {
    ESP_LOGI(TAG, "Current LED state: %s", getLedStateName(currentLedState));
}

// callback funk.
void handleSingleClick() {
    ESP_LOGI(TAG, "Single click detected - Setting LED to BLINK");
    if (ledPtr != NULL) {
        currentLedState = LedStatus::BLINK;
        ledPtr->setLedState(currentLedState);
        printCurrentState();
    }
}

void handleDoubleClick() {
    ESP_LOGI(TAG, "Double click detected - Setting LED to FAST_BLINK");
    if (ledPtr != NULL) {
        currentLedState = LedStatus::FAST_BLINK;
        ledPtr->setLedState(currentLedState);
        printCurrentState();
    }
}

void handleLongPress() {
    ESP_LOGI(TAG, "Long press detected - Setting LED to SLOW_BLINK");
    if (ledPtr != NULL) {
        currentLedState = LedStatus::SLOW_BLINK;
        ledPtr->setLedState(currentLedState);
        printCurrentState();
    }
}

void led_task(void *parameters) {
    ESP_LOGI(TAG, "Start LED Task");
    
    CLed *led;
    led = (CLed*)parameters;
    
    while(1) {
        led->tick();  
        vTaskDelay(10 / portTICK_PERIOD_MS);      
    }
}

// Task for Button
void button_task(void *parameters) {
    ESP_LOGI(TAG, "Start Button Task");
    
    CButton *button;
    button = (CButton*)parameters;
    
    while(1) {
        button->tick(); 
        //int buttonState = gpio_get_level((gpio_num_t)BUTTON_GPIO);
        //ESP_LOGI(TAG, "Button state: %d", buttonState);
        // nes za logiranje ne sjecam se sta
        vTaskDelay(5 / portTICK_PERIOD_MS);      
    }
}

// Task handles
TaskHandle_t ledTaskHandle = NULL;
TaskHandle_t buttonTaskHandle = NULL;

// ESP32 main function
extern "C" void app_main(void) {
    ESP_LOGI(TAG, "Start MAIN");
    
    CLed led(BLINK_GPIO);
    CButton button(BUTTON_GPIO);
    
    ledPtr = &led;
    
    currentLedState = LedStatus::OFF;
    led.setLedState(currentLedState);
    ESP_LOGI(TAG, "Initial LED state: %s", getLedStateName(currentLedState));
    
    button.attachSingleClick(handleSingleClick);
    button.attachDoubleClick(handleDoubleClick);
    button.attachLongPress(handleLongPress);
    
    xTaskCreate(led_task, "ledTask", 1024*5, (void*)&led, 1, &ledTaskHandle);
    xTaskCreate(button_task, "buttonTask", 1024*5, (void*)&button, 2, &buttonTaskHandle);
    
    ESP_LOGI(TAG, "Tasks Created"); 
    ESP_LOGI(TAG, "Press the button to change LED state:"); 
    ESP_LOGI(TAG, "- Single click: BLINK mode"); 
    ESP_LOGI(TAG, "- Double click: FAST_BLINK mode"); 
    ESP_LOGI(TAG, "- Long press: SLOW_BLINK mode"); 
    
    // Main loop
    while(1) {
        // The main functionality is now handled by tasks and callbacks
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
}