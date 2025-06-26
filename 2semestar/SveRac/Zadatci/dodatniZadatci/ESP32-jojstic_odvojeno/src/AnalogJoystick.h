#ifndef ANALOG_JOYSTICK_H
#define ANALOG_JOYSTICK_H

#include <stdint.h>
#include "driver/adc.h"

class AnalogJoystick {
private:
    adc1_channel_t vert_channel;
    adc1_channel_t horz_channel;

public:
    AnalogJoystick();
    void init();
    int read_vertical();
    int read_horizontal();
    void print_values();
};

#endif