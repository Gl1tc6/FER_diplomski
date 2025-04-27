#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <tuple>
#include <inttypes.h>

#include "esp_timer.h"
#include "esp_log.h"
#include "driver/gpio.h"

#include <freertos/FreeRTOS.h> 
#include "ets_sys.h"
#include "esp_idf_lib_helpers.h"


#include "./Util.hpp"

static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
#define PORT_ENTER_CRITICAL() portENTER_CRITICAL(&mux)
#define PORT_EXIT_CRITICAL() portEXIT_CRITICAL(&mux)

esp_err_t awaitState(gpio_num_t pin, int timeout_us, int expected_pin_state) {
	for (uint32_t i = 0; i < timeout_us; i += 10) {
		// need to wait at least a single interval to prevent reading a jitter
		ets_delay_us(10);
		if (gpio_get_level(pin) == expected_pin_state) {
			return ESP_OK;
		}
	}

	return ESP_ERR_TIMEOUT;
}

struct Fer2Sensor {
	gpio_num_t trigger;
	gpio_num_t data;
	
	void init() {
		gpio_set_direction(self.trigger, GPIO_MODE_OUTPUT);
		gpio_set_direction(self.data, GPIO_MODE_INPUT);
	}
	
	esp_err_t getMeasurement(int* temperature, int* humidity) {
		PORT_ENTER_CRITICAL();
		
		gpio_set_level(self.trigger, 0);
		gpio_set_level(self.trigger, 1);
		ets_delay_us(100);
		gpio_set_level(self.trigger, 0);
		
		esp_check self.readValue(temperature, 46);
		ets_delay_us(900);
		esp_check self.readValue(humidity, 100);
		
		PORT_EXIT_CRITICAL();
		return ESP_OK;
	}
	esp_err_t readValue(int* value, int max) {
		int val = 0;
		for (int i = 0; i < max; ++i) {
			esp_err_t r;
			r = awaitState(self.data, 100, 1);
			if (r == ESP_OK) {
				++val;
			} else if (r == ESP_ERR_TIMEOUT) {
				break;
			} else {
				ESP_ERROR_CHECK(r);
			}
			esp_check awaitState(self.data, 100, 0);
		}
		*value = val;
		return ESP_OK;
	}
};
