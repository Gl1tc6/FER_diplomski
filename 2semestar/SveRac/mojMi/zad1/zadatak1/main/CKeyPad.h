// CKeyPad.h
#pragma once

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include <functional>

// Definicije priključaka
#define C1 32
#define C2 27
#define C3 26
#define R1 12
#define R2 14
#define R3 25
#define R4 33

// Vremenske konstante
#define DEBOUNCE_TIME 20       // ms
#define DOUBLE_CLICK_TIME 250  // ms
#define LONG_PRESS_TIME 500    // ms
#define SCAN_INTERVAL 10       // ms

class CKeyPad {
public:
    // Definicije tipova povratnih funkcija
    typedef std::function<void(char)> CallbackFunction;

    CKeyPad();
    ~CKeyPad();

    // Metode za postavljanje povratnih funkcija
    void setSinglePressCallback(CallbackFunction callback);
    void setDoublePressCallback(CallbackFunction callback);
    void setLongPressCallback(CallbackFunction callback);

    // Inicijalizacija i start
    void init();
    void start();
    void stop();

private:
    // Row GPIO pinovi
    const gpio_num_t rows[4] = {
        static_cast<gpio_num_t>(R1),
        static_cast<gpio_num_t>(R2),
        static_cast<gpio_num_t>(R3),
        static_cast<gpio_num_t>(R4)
    };

    // Column GPIO pinovi
    const gpio_num_t cols[3] = {
        static_cast<gpio_num_t>(C1),
        static_cast<gpio_num_t>(C2),
        static_cast<gpio_num_t>(C3)
    };

    // Mapa tipkovnice
    const char keymap[4][3] = {
        {'1', '2', '3'},
        {'4', '5', '6'},
        {'7', '8', '9'},
        {'*', '0', '#'}
    };

    CallbackFunction singlePressCallback;
    CallbackFunction doublePressCallback;
    CallbackFunction longPressCallback;

    TaskHandle_t taskHandle;
    bool running;

    // Za praćenje stanja tipki
    char lastKey;
    uint32_t keyPressTime;
    uint32_t keyReleaseTime;
    bool longDetected;
    bool waitingForDouble;

    // Privatne metode
    void configureGPIO();
    char scanKeypad();
    static void keypadTask(void* arg);
    void processKeypad();
};