#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <tuple>
#include <inttypes.h>

#include "esp_timer.h"
#include "esp_log.h"
#include "driver/gpio.h"

#include "freertos/task.h"
#include <freertos/FreeRTOS.h> 
#include "ets_sys.h"
#include "esp_idf_lib_helpers.h"


#include "./Util.hpp"

void setToOutput(gpio_num_t pin) {
	gpio_set_direction(pin, GPIO_MODE_OUTPUT);
	gpio_pullup_en(pin);
}

int const pow10[2] {
	1, 
	10,
};

#define SVS_A (1 << 0)
#define SVS_B (1 << 1)
#define SVS_C (1 << 2)
#define SVS_D (1 << 3)
#define SVS_E (1 << 4)
#define SVS_F (1 << 5)
#define SVS_G (1 << 6)

uint8_t number_masks[10] {
	SVS_A | SVS_B | SVS_C | SVS_D | SVS_E | SVS_F,
	SVS_B | SVS_C,
	SVS_A | SVS_B | SVS_G | SVS_E | SVS_D,
	SVS_A | SVS_B | SVS_G | SVS_C | SVS_D,
	SVS_F | SVS_G | SVS_B | SVS_C,
	SVS_A | SVS_F | SVS_G | SVS_C | SVS_D,
	SVS_A | SVS_F | SVS_G | SVS_C | SVS_D | SVS_E,
	SVS_A | SVS_B | SVS_C,
	SVS_A | SVS_B | SVS_C | SVS_D | SVS_E | SVS_F | SVS_G,
	SVS_G | SVS_F | SVS_A | SVS_B | SVS_C | SVS_D,
};

struct MySvs {
	gpio_num_t digits[2];
	gpio_num_t segments[7];
	
	uint8_t value = 0;
	uint8_t current_digit = 0;
	
	void init() {
		for (auto pin : self.digits) setToOutput(pin);
		for (auto pin : self.segments) setToOutput(pin);
	}
	
	void tick() {
		int value = self.value; // copy just in case
		gpio_set_level(self.digits[self.current_digit], 0);
		self.current_digit += 1;
		self.current_digit %= 2;
		
		uint8_t digit_value = (value / pow10[self.current_digit]) % 10;
		for (int i = 0; i < 7; ++i) {
			gpio_set_level(self.segments[i], !((number_masks[digit_value] >> i) & 1));
		}
		
		gpio_set_level(self.digits[self.current_digit], 1);
		vTaskDelay(pdMS_TO_TICKS(10));
	}
};
