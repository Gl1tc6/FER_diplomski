#include "wokwi-api.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
  pin_t pin_out;
  uint32_t  voltage_attr;
  uint32_t  frequency_attr;
  uint32_t counter;
  uint32_t clk;
} chip_state_t; 

static void chip_timer_event(void *user_data);

void chip_init(void) {
  chip_state_t *chip = malloc(sizeof(chip_state_t));
  chip->pin_out = pin_init("OUT", ANALOG);
  chip->voltage_attr = attr_init_float("voltage", 5.0);
  chip->frequency_attr = attr_init_float("frequency", 50);

  const timer_config_t timer_config = {
    .callback = chip_timer_event,
    .user_data = chip,
  };
  timer_t timer_id = timer_init(&timer_config);
  timer_start(timer_id, 100, true);
}

void chip_timer_event(void *user_data) {
  chip_state_t *chip = (chip_state_t*)user_data;

  //Counte every 100us
  chip->counter++;  
  
  //2.5 * sin( f * 2 * pi * t) * 2.5
  float f = attr_read_float(chip->frequency_attr);
  float v = attr_read_float (chip->voltage_attr);  
  float t = chip->counter / 10000.0;
  
  float voltage = 2.5 * sin(f * 2 * 3.1415926535 * t) + 2.5;
  //printf("%f %f %f %f\n", voltage, f, v, t);

  pin_dac_write(chip->pin_out, voltage);
}