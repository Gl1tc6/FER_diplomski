#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/adc.h"
#include "esp_adc_cal.h"
#include "esp_timer.h"

#define ADC_CHANNEL ADC1_CHANNEL_0
#define ADC_ATTEN ADC_ATTEN_DB_12
#define ADC_WIDTH ADC_WIDTH_BIT_12
#define ADC_VREF 1100

static esp_adc_cal_characteristics_t *adc_chars;
static volatile bool frequency_measurement_done = false;
static volatile float measured_frequency = 0.0;
static volatile int zero_crossings = 0;

void init_adc(void)
{
    adc1_config_width(ADC_WIDTH);
    adc1_config_channel_atten(ADC_CHANNEL, ADC_ATTEN);
    
    adc_chars = (esp_adc_cal_characteristics_t*)calloc(1, sizeof(esp_adc_cal_characteristics_t));
    esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN, ADC_WIDTH, ADC_VREF, adc_chars);
}

float read_voltage(void)
{
    uint32_t adc_reading = 0, pom;
    for (int i = 0; i < 64; i++) {
        pom = adc1_get_raw(ADC_CHANNEL);
        adc_reading += pom;
        printf("%ld\n", pom);
        }

    adc_reading /= 64;

    uint32_t voltage = esp_adc_cal_raw_to_voltage(adc_reading, adc_chars);
    return voltage  / 1000.0;
}

float calculate_effective_voltage(float input_voltage)
{
    if (input_voltage < 4.0 || input_voltage > 5.0) {
        return 0.0;
    }
    
    float u_max = 310.0 + 30.0 * (input_voltage - 4.0);
    return u_max / sqrt(2.0);
}

void measure_frequency_task(void *pvParameters)
{
    float voltage_samples[20];
    int sample_index = 0;
    float dc_offset = 4.5;
    bool previous_above_offset = false;
    bool first_sample = true;
    int64_t first_crossing_time = 0;
    int64_t last_crossing_time = 0;
    bool first_crossing_detected = false;
    int crossing_count = 0;
    
    while (1) {
        if (frequency_measurement_done) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        
        uint32_t adc_raw = adc1_get_raw(ADC_CHANNEL);
        uint32_t voltage_mv = esp_adc_cal_raw_to_voltage(adc_raw, adc_chars);
        float current_voltage = voltage_mv / 1000.0;
        
        voltage_samples[sample_index] = current_voltage;
        sample_index = (sample_index + 1) % 20;
        
        if (sample_index == 0) {
            float sum = 0;
            for (int i = 0; i < 20; i++) {
                sum += voltage_samples[i];
            }
            dc_offset = sum / 20.0;
        }
        
        bool current_above_offset = (current_voltage > dc_offset);
        
        if (!first_sample && (previous_above_offset != current_above_offset)) {
            int64_t current_time = esp_timer_get_time();
            
            if (!first_crossing_detected) {
                first_crossing_time = current_time;
                first_crossing_detected = true;
                crossing_count = 1;
            } else {
                crossing_count++;
                last_crossing_time = current_time;
                
                if (crossing_count >= 10) {
                    float time_diff_s = (last_crossing_time - first_crossing_time) / 1000000.0;
                    float half_periods = crossing_count - 1;
                    
                    if (time_diff_s > 0) {
                        measured_frequency = half_periods / (2.0 * time_diff_s);
                        zero_crossings = crossing_count;
                        frequency_measurement_done = true;
                    }
                    
                    first_crossing_detected = false;
                    crossing_count = 0;
                }
            }
        }
        
        previous_above_offset = current_above_offset;
        first_sample = false;
        
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

void app_main(void)
{
    printf("Power Monitor ESP32 Started\n");
    
    init_adc();
    
    xTaskCreate(measure_frequency_task, "freq_task", 4096, NULL, 5, NULL);
    
    while (1) {
        frequency_measurement_done = false;
        
        int timeout = 0;
        while (!frequency_measurement_done && timeout < 300) {
            vTaskDelay(pdMS_TO_TICKS(10));
            timeout++;
        }
        
        float input_voltage = read_voltage();
        float effective_voltage = calculate_effective_voltage(input_voltage);
        
        printf("\n=== MJERENJE ===\n");
        printf("Ulazni napon: %.3f V\n", input_voltage);
        
        if (effective_voltage > 0) {
            printf("Efektivni napon: %.1f V\n", effective_voltage);
        } else {
            printf("Efektivni napon: GRESKA\n");
        }
        
        if (frequency_measurement_done && measured_frequency > 0) {
            printf("Frekvencija: %.2f Hz\n", measured_frequency);
        } else {
            printf("Frekvencija: GRESKA\n");
        }
        
        printf("Prijelazi: %d\n", (int)zero_crossings);
        printf("================\n");
        
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}