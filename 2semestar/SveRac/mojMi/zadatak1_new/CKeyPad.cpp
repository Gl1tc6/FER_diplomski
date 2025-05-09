// CKeyPad.cpp
#include "CKeyPad.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include <stdio.h>

static const char* TAG = "CKeyPad";

CKeyPad::CKeyPad() : 
    singlePressCallback(nullptr),
    doublePressCallback(nullptr),
    longPressCallback(nullptr),
    taskHandle(nullptr),
    running(false),
    lastKey(0),
    keyPressTime(0),
    keyReleaseTime(0),
    longDetected(false),
    waitingForDouble(false),
    lastReleasedKey(0) {  // Add this new member variable initialization
}

CKeyPad::~CKeyPad() {
    stop();
}

void CKeyPad::setSinglePressCallback(CallbackFunction callback) {
    singlePressCallback = callback;
}

void CKeyPad::setDoublePressCallback(CallbackFunction callback) {
    doublePressCallback = callback;
}

void CKeyPad::setLongPressCallback(CallbackFunction callback) {
    longPressCallback = callback;
}

void CKeyPad::init() {
    configureGPIO();
}

void CKeyPad::start() {
    if (!running) {
        running = true;
        xTaskCreate(keypadTask, "keypad_task", 4096, this, 10, &taskHandle);
    }
}

void CKeyPad::stop() {
    if (running) {
        running = false;
        if (taskHandle != nullptr) {
            vTaskDelete(taskHandle);
            taskHandle = nullptr;
        }
    }
}

void CKeyPad::configureGPIO() {
    // Konfiguracija row pinova kao izlaza
    for (int i = 0; i < 4; i++) {
        gpio_config_t io_conf = {};
        io_conf.intr_type = GPIO_INTR_DISABLE;
        io_conf.mode = GPIO_MODE_OUTPUT;
        io_conf.pin_bit_mask = (1ULL << rows[i]);
        io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
        io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
        gpio_config(&io_conf);
        gpio_set_level(rows[i], 1); // Postavimo sve na HIGH inicijalno
    }

    // Konfiguracija column pinova kao ulaza s PULL-UP
    for (int i = 0; i < 3; i++) {
        gpio_config_t io_conf = {};
        io_conf.intr_type = GPIO_INTR_DISABLE;
        io_conf.mode = GPIO_MODE_INPUT;
        io_conf.pin_bit_mask = (1ULL << cols[i]);
        io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
        io_conf.pull_up_en = GPIO_PULLUP_ENABLE;
        gpio_config(&io_conf);
    }
}

char CKeyPad::scanKeypad() {
    // Petlja kroz redove (ROW)
    for (int r = 0; r < 4; r++) {
        // Postavimo trenutni red na 0 (LOW), a sve ostale na 1 (HIGH)
        for (int i = 0; i < 4; i++) {
            gpio_set_level(rows[i], i != r);
        }
        
        // Mala pauza za stabilizaciju signala
        vTaskDelay(1 / portTICK_PERIOD_MS);
        
        // Provjera stupaca (COL)
        for (int c = 0; c < 3; c++) {
            if (gpio_get_level(cols[c]) == 0) {
                // Tipka je pritisnuta (detektiramo LOW na stupcu)
                return keymap[r][c];
            }
        }
    }
    
    // Nijedna tipka nije pritisnuta
    return 0;
}

void CKeyPad::keypadTask(void* arg) {
    CKeyPad* keypad = static_cast<CKeyPad*>(arg);
    
    while (keypad->running) {
        keypad->processKeypad();
        vTaskDelay(SCAN_INTERVAL / portTICK_PERIOD_MS);
    }
    
    vTaskDelete(nullptr);
}

void CKeyPad::processKeypad() {
    char key = scanKeypad();
    uint32_t currentTime = xTaskGetTickCount() * portTICK_PERIOD_MS;
    
    // Ako je tipka pritisnuta
    if (key != 0) {
        // Ako je ovo novi pritisak (nije ista tipka kao ranije)
        if (lastKey != key) {
            lastKey = key;
            keyPressTime = currentTime;
            longDetected = false;
        } 
        // Provjera za long press
        else if (!longDetected && (currentTime - keyPressTime) > LONG_PRESS_TIME) {
            if (longPressCallback) {
                longPressCallback(key);
            }
            longDetected = true;
            waitingForDouble = false; // Poništavamo čekanje na dvostruki klik
        }
    } 
    // Ako nijedna tipka nije pritisnuta, ali je ranije bila pritisnuta
    else if (lastKey != 0) {
        keyReleaseTime = currentTime;
        lastReleasedKey = lastKey;  // Store the released key
        lastKey = 0;
        
        // Ako nije bio long press i ne čekamo drugi klik za double click
        if (!longDetected && !waitingForDouble) {
            waitingForDouble = true;
            // Počinjemo čekati za drugi klik (double click)
            vTaskDelay(5 / portTICK_PERIOD_MS); // Mali delay da izbjegnemo bouncing
        } 
        // Provjera za double click
        else if (waitingForDouble) {
            waitingForDouble = false;
            
            // Ako je prošlo dovoljno vremena za debounce ali ne previše za double click
            if ((currentTime - keyReleaseTime) > DEBOUNCE_TIME && 
                (currentTime - keyPressTime) < DOUBLE_CLICK_TIME) {
                if (doublePressCallback) {
                    doublePressCallback(lastReleasedKey);  // Use the stored released key
                }
            } else {
                // Ovo je single press
                if (singlePressCallback && !longDetected) {
                    singlePressCallback(lastReleasedKey);  // Use the stored released key
                }
            }
        }
    } 
    // Provjera isteka vremena za dvostruki klik
    else if (waitingForDouble && (currentTime - keyReleaseTime) > DOUBLE_CLICK_TIME) {
        waitingForDouble = false;
        // Ovo je bio single press
        if (singlePressCallback && !longDetected) {
            singlePressCallback(lastReleasedKey);  // Use the stored released key
        }
    }
}