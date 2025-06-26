#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/adc.h"
#include "esp_adc_cal.h"

// Definisanje pinova za joystick
#define JOYSTICK_VERT_PIN ADC1_CHANNEL_6  // GPIO34
#define JOYSTICK_HORZ_PIN ADC1_CHANNEL_7  // GPIO35

// ADC kalibracija
static esp_adc_cal_characteristics_t *adc_chars;

void adc_init() {
    // Konfiguracija ADC1
    adc1_config_width(ADC_WIDTH_BIT_9);  // 9-bit rezolucija (0-511)
    
    // Konfiguracija kanala za vertikalnu os
    adc1_config_channel_atten(JOYSTICK_VERT_PIN, ADC_ATTEN_DB_2_5);  // 100mV - 950mV
    
    // Konfiguracija kanala za horizontalnu os
    adc1_config_channel_atten(JOYSTICK_HORZ_PIN, ADC_ATTEN_DB_2_5);  // 100mV - 950mV
    
    // Alokacija memorije za ADC karakteristike
    adc_chars = (esp_adc_cal_characteristics_t*)calloc(1, sizeof(esp_adc_cal_characteristics_t));
    
    // Karakterizacija ADC-a
    esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_2_5, ADC_WIDTH_BIT_9, 1100, adc_chars);
    
    printf("ADC inicijalizovan uspješno\n");
}

void app_main() {
    // Inicijalizacija ADC-a
    adc_init();
    
    printf("ESP32 Analog Joystick Program\n");
    printf("Čitanje vrijednosti svakih 500ms...\n\n");
    
    while (1) {
        // Čitanje horizontalne vrijednosti (single read mode)
        int horz_raw = adc1_get_raw(JOYSTICK_HORZ_PIN);
        
        // Čitanje vertikalne vrijednosti (single read mode)  
        int vert_raw = adc1_get_raw(JOYSTICK_VERT_PIN);

        
        // Konvertovanje u napon
        //uint32_t horz_voltage = esp_adc_cal_raw_to_voltage(horz_raw, adc_chars);
        //uint32_t vert_voltage = esp_adc_cal_raw_to_voltage(vert_raw, adc_chars);

        float horz_voltage = (float)horz_raw / 511.0 * 850.0 + 100.0;
        float vert_voltage = (float)vert_raw / 511.0 * 850.0 + 100.0;

        // Ispis rezultata preko serijskog porta
        printf("Horizontalna os: %d (raw) = %.2f mV\n", horz_raw, horz_voltage);
        printf("Vertikalna os:   %d (raw) = %.2f mV\n", vert_raw, vert_voltage);
        printf("------------------------\n");
        
        // Čekanje 500ms
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}