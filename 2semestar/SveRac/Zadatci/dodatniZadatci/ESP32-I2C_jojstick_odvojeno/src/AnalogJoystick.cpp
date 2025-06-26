#include "AnalogJoystick.h"
#include "driver/adc.h"
#include "esp_log.h"
#include <stdio.h>

static const char* TAG = "JOYSTICK";

AnalogJoystick::AnalogJoystick() {
    vert_channel = ADC1_CHANNEL_6;  // GPIO34
    horz_channel = ADC1_CHANNEL_7;  // GPIO35
}

void AnalogJoystick::init() {
    // Konfiguracija ADC1
    adc1_config_width(ADC_WIDTH_BIT_9);
    adc1_config_channel_atten(vert_channel, ADC_ATTEN_DB_2_5);
    adc1_config_channel_atten(horz_channel, ADC_ATTEN_DB_2_5);
    
    ESP_LOGI(TAG, "ADC initialized - 9-bit resolution, 100mV-950mV range");
}

int AnalogJoystick::read_vertical() {
    return adc1_get_raw(vert_channel);
}

int AnalogJoystick::read_horizontal() {
    return adc1_get_raw(horz_channel);
}

void AnalogJoystick::print_values() {
    int vert_val = read_vertical();
    int horz_val = read_horizontal();
    
    ESP_LOGI(TAG, "VERT: %d, HORZ: %d", vert_val, horz_val);
}