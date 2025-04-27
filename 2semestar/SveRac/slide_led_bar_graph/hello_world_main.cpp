#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
// #include "driver/adc.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"

#include "./MyVoltage.hpp"

#define MIN_MV 142
#define MAX_MV 3171

gpio_num_t const led_strip[10] {
	GPIO_NUM_0, GPIO_NUM_4, GPIO_NUM_16, GPIO_NUM_17, GPIO_NUM_5, GPIO_NUM_18, GPIO_NUM_19, GPIO_NUM_21, GPIO_NUM_22, GPIO_NUM_23, 
};

extern "C" void app_main() {
	MyAdc adc(ADC_UNIT_1);
	MyChannel channel(adc, ADC_CHANNEL_6, ADC_ATTEN_DB_12, ADC_BITWIDTH_12);
	MyVoltage voltager(channel);
	
	for (auto led : led_strip) {
		gpio_set_direction(led, GPIO_MODE_OUTPUT);
		gpio_pullup_en(led);
	}
	
	while (true) {
		int voltage_mV = voltager.get_mV();
		printf("voltage: %d mV ", voltage_mV);
		voltage_mV -= MIN_MV;
		int led_count = voltage_mV * 11 / (MAX_MV - MIN_MV);
		printf("led count: %d\n", led_count);
		for (int i = 0; i < 10; ++i) {
			gpio_set_level(led_strip[i], i < led_count);
		}
		
		vTaskDelay(pdMS_TO_TICKS(100));
		// vTaskDelay(pdMS_TO_TICKS(1000));
	}
	
}


