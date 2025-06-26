#include "ckeypad.h"
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_log.h"

static const char TAG[] = "CKeyPad";

// Forward declaration of the scanning task
static void keypad_scan_task(void* param);

void ckeypad_init(CKeyPad* keypad) {
    // Initialize row pins (outputs)
    keypad->row_pins[0] = R1;
    keypad->row_pins[1] = R2;
    keypad->row_pins[2] = R3;
    keypad->row_pins[3] = R4;
    
    // Initialize column pins (inputs with pull-up)
    keypad->col_pins[0] = C1;
    keypad->col_pins[1] = C2;
    keypad->col_pins[2] = C3;
    
    // Initialize keypad layout
    keypad->key_map[0][0] = '1'; keypad->key_map[0][1] = '2'; keypad->key_map[0][2] = '3';
    keypad->key_map[1][0] = '4'; keypad->key_map[1][1] = '5'; keypad->key_map[1][2] = '6';
    keypad->key_map[2][0] = '7'; keypad->key_map[2][1] = '8'; keypad->key_map[2][2] = '9';
    keypad->key_map[3][0] = '*'; keypad->key_map[3][1] = '0'; keypad->key_map[3][2] = '#';
    
    // Initialize callbacks to NULL
    keypad->single_press_callback = NULL;
    keypad->double_press_callback = NULL;
    keypad->long_press_callback = NULL;
    
    // Initialize state
    keypad->last_key = 0;
    keypad->previous_key = 0;
    keypad->last_press_time = 0;
    keypad->key_press_start_time = 0;
    keypad->key_pressed = false;
    keypad->waiting_for_double = false;
    keypad->long_press_triggered = false;
    
    // Initialize timing constants (in microseconds)
    keypad->long_press_threshold = 500000;  // 500ms
    keypad->double_press_window = 300000;   // 300ms
    keypad->scan_interval = 10000;          // 10ms
    
    keypad->task_handle = NULL;
    keypad->running = false;
    
    // Configure GPIO pins
    for (int i = 0; i < KEYPAD_ROWS; i++) {
        gpio_set_direction(keypad->row_pins[i], GPIO_MODE_OUTPUT);
        gpio_set_level(keypad->row_pins[i], 1);
    }
    
    for (int i = 0; i < KEYPAD_COLS; i++) {
        gpio_set_direction(keypad->col_pins[i], GPIO_MODE_INPUT);
        gpio_set_pull_mode(keypad->col_pins[i], GPIO_PULLUP_ONLY);
    }
    
    ESP_LOGI(TAG, "CKeyPad initialized");
}

void ckeypad_set_single_press_callback(CKeyPad* keypad, keypad_callback_t callback) {
    keypad->single_press_callback = callback;
}

void ckeypad_set_double_press_callback(CKeyPad* keypad, keypad_callback_t callback) {
    keypad->double_press_callback = callback;
}

void ckeypad_set_long_press_callback(CKeyPad* keypad, keypad_callback_t callback) {
    keypad->long_press_callback = callback;
}

char ckeypad_scan(CKeyPad* keypad) {
    // Scan patterns: 0111, 1011, 1101, 1110
    uint8_t row_patterns[KEYPAD_ROWS] = {0x0E, 0x0D, 0x0B, 0x07}; // 0111, 1011, 1101, 1110
    
    for (int row = 0; row < KEYPAD_ROWS; row++) {
        // Set row pattern
        for (int i = 0; i < KEYPAD_ROWS; i++) {
            gpio_set_level(keypad->row_pins[i], (row_patterns[row] >> i) & 1);
        }
        
        // Small delay for signal to settle
        vTaskDelay(pdMS_TO_TICKS(1));
        
        // Read columns
        for (int col = 0; col < KEYPAD_COLS; col++) {
            if (gpio_get_level(keypad->col_pins[col]) == 0) {
                // Key is pressed (pull-up resistor makes it 0 when pressed)
                return keypad->key_map[row][col];
            }
        }
    }
    
    return 0; // No key pressed
}

static void keypad_scan_task(void* param) {
    CKeyPad* keypad = (CKeyPad*)param;
    char current_key;
    uint64_t current_time;
    
    while (keypad->running) {
        current_key = ckeypad_scan(keypad);
        current_time = esp_timer_get_time();
        
        if (current_key != 0) {
            // Key is pressed
            if (!keypad->key_pressed) {
                // New key press detected
                keypad->key_pressed = true;
                keypad->key_press_start_time = current_time;
                keypad->last_key = current_key;
                keypad->long_press_triggered = false;
                
                ESP_LOGD(TAG, "Key pressed: %c", current_key);
            } else {
                // Key is still being pressed, check for long press
                if (!keypad->long_press_triggered && 
                    (current_time - keypad->key_press_start_time) >= keypad->long_press_threshold) {
                    
                    keypad->long_press_triggered = true;
                    if (keypad->long_press_callback) {
                        keypad->long_press_callback(keypad->last_key, PRESS_LONG);
                    }
                    ESP_LOGI(TAG, "Long press detected: %c", keypad->last_key);
                }
            }
        } else {
            // No key is pressed
            if (keypad->key_pressed) {
                // Key was just released
                keypad->key_pressed = false;
                
                if (!keypad->long_press_triggered) {
                    // It was not a long press, check for single or double press
                    if (keypad->waiting_for_double && 
                        keypad->last_key == keypad->previous_key && 
                        (current_time - keypad->last_press_time) <= keypad->double_press_window) {
                        
                        // Double press detected
                        keypad->waiting_for_double = false;
                        if (keypad->double_press_callback) {
                            keypad->double_press_callback(keypad->last_key, PRESS_DOUBLE);
                        }
                        ESP_LOGI(TAG, "Double press detected: %c", keypad->last_key);
                    } else {
                        // Start waiting for potential double press
                        keypad->waiting_for_double = true;
                        keypad->previous_key = keypad->last_key;
                        keypad->last_press_time = current_time;
                    }
                }
            }
            
            // Check if double press window has expired
            if (keypad->waiting_for_double && 
                (current_time - keypad->last_press_time) > keypad->double_press_window) {
                
                keypad->waiting_for_double = false;
                if (keypad->single_press_callback) {
                    keypad->single_press_callback(keypad->previous_key, PRESS_SINGLE);
                }
                ESP_LOGI(TAG, "Single press detected: %c", keypad->previous_key);
            }
        }
        
        vTaskDelay(pdMS_TO_TICKS(10)); // 10ms scan interval
    }
    
    vTaskDelete(NULL);
}

void ckeypad_start(CKeyPad* keypad) {
    if (!keypad->running) {
        keypad->running = true;
        xTaskCreate(keypad_scan_task, "keypad_scan", 2048, keypad, 10, (TaskHandle_t*)&keypad->task_handle);
        ESP_LOGI(TAG, "CKeyPad scanning started");
    }
}

void ckeypad_stop(CKeyPad* keypad) {
    if (keypad->running) {
        keypad->running = false;
        ESP_LOGI(TAG, "CKeyPad scanning stopped");
    }
}