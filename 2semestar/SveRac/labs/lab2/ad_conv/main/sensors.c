#include <stdio.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/adc.h"
#include "esp_adc_cal.h"
#include "math.h"
#include "esp_log.h"
#include "esp_timer.h"

#define DHT_PIN GPIO_NUM_15
static esp_adc_cal_characteristics_t *adc_chars;

void init(void){

    // DHT
    gpio_set_pull_mode(DHT_PIN, GPIO_PULLUP_ONLY);

    // ADC
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC_CHANNEL_6,ADC_ATTEN_DB_11);
    adc_chars = calloc(1,sizeof(esp_adc_cal_characteristics_t));
    esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11,ADC_WIDTH_BIT_12, 1100, adc_chars);
}

float dht_temp(void) {
    uint8_t data[5] = {0};
    
    // timers set by this datasheet: https://cdn.sparkfun.com/assets/f/7/d/9/c/DHT22.pdf
    // wake sensor
    gpio_set_direction(DHT_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(DHT_PIN, 0);
    esp_rom_delay_us(1050);
    gpio_set_level(DHT_PIN, 1);
    gpio_set_direction(DHT_PIN, GPIO_MODE_INPUT);
    esp_rom_delay_us(30);

    if (gpio_get_level(DHT_PIN) != 0)
    {
        ESP_LOGI("DHT22_s", "timeout");
        return -1000;
    }
    
    esp_rom_delay_us(160);
    
    // data
    for (int i = 0; i < 40; i++) {
        int64_t timeout = esp_timer_get_time() + 70;
        while (gpio_get_level(DHT_PIN) == 0) {
            if (esp_timer_get_time() > timeout) {
                return NAN;
            }
        }
        
        int64_t start = esp_timer_get_time();
        timeout = start + 100;
        
        while (gpio_get_level(DHT_PIN) == 1) {
            if (esp_timer_get_time() > timeout) {
                return NAN;
            }
        }
        
        if (esp_timer_get_time() - start > 50) {
            uint8_t B_idx = i / 8;  
            uint8_t b_pos = 7 - (i % 8);
            data[B_idx] |= (1 << b_pos);
        }
    }
    
    // temp
    uint16_t temp_raw = ((data[2] & 0x7F) << 8) | data[3];
    float temp = temp_raw * 0.1f;
    
    if (data[2] & 0x80) {
        temp = -temp;
    }
    
    return temp;
}



float adc_temp(void){
    uint32_t adc_reading = 0;
    adc_reading =adc1_get_raw((adc1_channel_t)ADC_CHANNEL_6);
    ESP_LOGI("adc_read", "%ld", adc_reading);
    // //Convert adc_reading to voltage in mV -> V
    float voltage =esp_adc_cal_raw_to_voltage(adc_reading, adc_chars)/1000.0f;
    ESP_LOGI("VMA320_s", "Voltage:\t%f V       ", voltage);

    const float BETA = 3950; // 
    
    float temp = 1 / (log(1 / (4095. / adc_reading - 1)) / BETA + 1.0 / 298.15) - 273.15;
    return temp;
}