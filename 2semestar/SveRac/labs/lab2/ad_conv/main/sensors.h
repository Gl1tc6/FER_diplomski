#include "esp_err.h"
#include "driver/gpio.h"
#include "driver/adc.h"

// Initialize sensor hardware
void init(void);

float dht_temp(void);

float adc_temp(void);