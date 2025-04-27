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


#include "Fer2Sensor.hpp"
#include "SvS.hpp"


#define DATA_IN GPIO_NUM_4
#define DATA_OUT GPIO_NUM_2
#define A GPIO_NUM_19
#define B GPIO_NUM_21
#define C GPIO_NUM_32
#define D GPIO_NUM_33
#define E GPIO_NUM_25
#define F GPIO_NUM_27
#define G GPIO_NUM_26
#define DIG1 GPIO_NUM_14
#define DIG2 GPIO_NUM_12 


// MyVoltage* voltager;
// void printMeasurements() {
// 	int voltage_mV = voltager->get_mV();
// 	int resistance_Ohm = getResistance_Ohm<10 * 1000, 5000>(voltage_mV); // R1 = 10 kOhm; Vin = 5 V
// 	auto analog_temperature_oC = getTemperature<TemperatureUnitId::CELSIUS>(resistance_Ohm);
// 	float dht_temperature, hum;
// 	esp_check dht_read_float_data(DHT_TYPE_AM2301, GPIO_NUM_15, &hum, &dht_temperature);
// 	printf("VMA: %f °C    DHT: %f °C\n", analog_temperature_oC.value, dht_temperature);
// }

Fer2Sensor f2sensor{ GPIO_NUM_4, GPIO_NUM_2 };
MySvs svs{
	{DIG2, DIG1},
	{A, B, C, D, E, F, G},
};

void svsTask(void* _) {
	while (true) {
		svs.tick();
	}
}

extern "C" void app_main() {
	f2sensor.init();
	svs.init();
	
	// MyAdc adc(ADC_UNIT_1);
	// MyChannel channel(adc, ADC_CHANNEL_6, ADC_ATTEN_DB_12, ADC_BITWIDTH_12);
	// MyVoltage voltager(channel);
	// ::voltager = &voltager;
	
	// // esp_check gpio_reset_pin(GPIO_NUM_15);
	// // MyDht<TemperatureUnitId::CELSIUS> dht(GPIO_NUM_15);
	// esp_check gpio_reset_pin(GPIO_NUM_15);
	// esp_check gpio_set_pull_mode(GPIO_NUM_15, GPIO_PULLUP_ONLY);
	
	// CButton button(GPIO_NUM_13);
	// button.attachSingleClick(printMeasurements);
	TaskHandle_t handle;
	xTaskCreate(
		svsTask,   // Task function
		"svs",                          // Name of task in task scheduler
		1024 * 5,                      // Stack size
		nullptr,                 // Parameter send to function
		1,                             // Priority
		&handle                         // task handler 
	);
	
	while (true) {
		int t, h;
		
		esp_check f2sensor.getMeasurement(&t, &h);
		svs.value = t;
		printf("T %d  %d \n", t, h);
		vTaskDelay(pdMS_TO_TICKS(1000));
		
		esp_check f2sensor.getMeasurement(&t, &h);
		svs.value = h;
		printf("H %d  %d \n", t, h);
		vTaskDelay(pdMS_TO_TICKS(1000));
		
		// vTaskDelay(pdMS_TO_TICKS(10));
	}
	
}


