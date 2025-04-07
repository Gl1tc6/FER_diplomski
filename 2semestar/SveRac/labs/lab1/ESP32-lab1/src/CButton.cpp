#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "CButton.h"

#define DEBOUNCE_TIME      50000L    // 50ms
#define DOUBLE_CLICK_TIME  300000L   // 300ms
#define LONG_PRESS_TIME    1000000L  // 1s
const char *LogName = "CButton";

CButton::CButton(int port){

    //ToDo::
    //BOOT is GPIO0 (HIGH when released, LOW when pressed)
    m_pinNumber = (gpio_num_t)port;
    ESP_LOGI(LogName, "Configure port[%d] as input for button", port);
    
    // GPIO 
    gpio_reset_pin(m_pinNumber);
    gpio_set_direction(m_pinNumber, GPIO_MODE_INPUT);
    gpio_set_pull_mode(m_pinNumber, GPIO_PULLUP_ONLY);
    gpio_set_intr_type(m_pinNumber, GPIO_INTR_POSEDGE); // rising edge
    
    m_lastState = 1;
    m_lastPressTime = 0;
    m_lastReleaseTime = 0;
    m_clickCount = 0;
    m_isLongPressDetected = false;
}

void CButton::tick(){
    //ToDo
    int currentState = gpio_get_level(m_pinNumber);
    int64_t currentTime = esp_timer_get_time();
    
    // za debounce
    if (currentState != m_lastState && (currentTime - m_lastStateChangeTime) > DEBOUNCE_TIME) {
        m_lastStateChangeTime = currentTime;
        
        if (currentState == 0) {
            m_lastPressTime = currentTime;
            m_isPressed = true;
            m_isLongPressDetected = false;
            ESP_LOGD(LogName, "Button pressed");
        } 
        else {
            m_lastReleaseTime = currentTime;
            m_isPressed = false;
            
            if (!m_isLongPressDetected) {
                m_clickCount++;
                ESP_LOGD(LogName, "Click count: %d", m_clickCount);
                
                // nastavi obrađivati
                m_processingClicks = true;
            }
        }
        m_lastState = currentState;
    }
    
    // long press ako lp već nije detektiran
    if (m_isPressed && !m_isLongPressDetected && (currentTime - m_lastPressTime) > LONG_PRESS_TIME) {
        m_isLongPressDetected = true;
        m_clickCount = 0;
        
        ESP_LOGI(LogName, "Long press detected");
        if (longPress != NULL) {
            longPress();
        }
    }
    
    // obradi nakon čekanja - detktiraj dupli klik
    if (m_processingClicks && !m_isPressed && (currentTime - m_lastReleaseTime) > DOUBLE_CLICK_TIME) {
        m_processingClicks = false;
        
        // obradi 1 / 2 klika
        if (m_clickCount == 1) {
            ESP_LOGI(LogName, "Single click detected");
            if (singleClick != NULL) {
                singleClick();
            }
        } else if (m_clickCount == 2) {
            ESP_LOGI(LogName, "Double click detected");
            if (doubleClick != NULL) {
                doubleClick();
            }
        }
        m_clickCount = 0;
    }
}
