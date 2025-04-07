#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "CButton.h"

#define DEBOUNCE_TIME      50000L    // 50ms in microseconds
#define DOUBLE_CLICK_TIME  300000L   // 300ms in microseconds
#define LONG_PRESS_TIME    1000000L  // 1s in microseconds


CButton::CButton(int port){

    //ToDo::
    //BOOT is GPIO0 (HIGH when released, LOW when pressed)
    m_pinNumber = (gpio_num_t)port;
    ESP_LOGI(LogName, "Configure port[%d] as input for button", port);
    
    // Configure GPIO
    gpio_reset_pin(m_pinNumber);
    gpio_set_direction(m_pinNumber, GPIO_MODE_INPUT);
    gpio_set_pull_mode(m_pinNumber, GPIO_PULLUP_ONLY); 
    gpio_set_intr_type(m_pinNumber, GPIO_INTR_POSEDGE); // rising edge
    
    // Initialize member variables
    m_lastState = 1;  // Buttons are typically HIGH when not pressed (with pull-up)
    m_lastPressTime = 0;
    m_lastReleaseTime = 0;
    m_clickCount = 0;
    m_isLongPressDetected = false;
}

void CButton::tick(){
    //ToDo
    int currentState = gpio_get_level(m_pinNumber);
    int64_t currentTime = esp_timer_get_time();
    
    // Check for state change with debounce
    if (currentState != m_lastState && (currentTime - m_lastStateChangeTime) > DEBOUNCE_TIME) {
        m_lastStateChangeTime = currentTime;
        
        // Button pressed (transition from HIGH to LOW)
        if (currentState == 0) {
            m_lastPressTime = currentTime;
            m_isPressed = true;
            m_isLongPressDetected = false;
            ESP_LOGD(LogName, "Button pressed");
        } 
        // Button released (transition from LOW to HIGH)
        else {
            m_lastReleaseTime = currentTime;
            m_isPressed = false;
            
            // If long press wasn't already detected
            if (!m_isLongPressDetected) {
                // Increment click count
                m_clickCount++;
                ESP_LOGD(LogName, "Click count: %d", m_clickCount);
                
                // Start timer to process clicks after a delay
                m_processingClicks = true;
            }
        }
        
        m_lastState = currentState;
    }
    
    // Check for long press if button is currently pressed and long press not yet detected
    if (m_isPressed && !m_isLongPressDetected && (currentTime - m_lastPressTime) > LONG_PRESS_TIME) {
        m_isLongPressDetected = true;
        m_clickCount = 0;  // Reset click count
        
        ESP_LOGI(LogName, "Long press detected");
        if (longPress != NULL) {
            longPress();
        }
    }
    
    // Process clicks after a delay to allow for double-click detection
    if (m_processingClicks && !m_isPressed && (currentTime - m_lastReleaseTime) > DOUBLE_CLICK_TIME) {
        m_processingClicks = false;
        
        // Process single or double click
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
        
        m_clickCount = 0;  // Reset click count
    }
}
