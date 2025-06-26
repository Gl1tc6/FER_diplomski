#include "SevenSegmentDisplay.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <stdio.h>

const uint8_t SevenSegmentDisplay::digit_to_segments[10] = {
    0b1000000, // 0: A,B,C,D,E,F (segments A,B,C,D,E,F on)
    0b1111001, // 1: B,C (segments B,C on)
    0b0100100, // 2: A,B,D,E,G (segments A,B,D,E,G on)
    0b0110000, // 3: A,B,C,D,G (segments A,B,C,D,G on)
    0b0011001, // 4: B,C,F,G (segments B,C,F,G on)
    0b0010010, // 5: A,C,D,F,G (segments A,C,D,F,G on)
    0b0000010, // 6: A,C,D,E,F,G (segments A,C,D,E,F,G on)
    0b1111000, // 7: A,B,C (segments A,B,C on)
    0b0000000, // 8: A,B,C,D,E,F,G (all segments on)
    0b0010000  // 9: A,B,C,D,F,G (segments A,B,C,D,F,G on)
};

SevenSegmentDisplay::SevenSegmentDisplay() {
    segment_pins[0] = SEG_A;
    segment_pins[1] = SEG_B;
    segment_pins[2] = SEG_C;
    segment_pins[3] = SEG_D;
    segment_pins[4] = SEG_E;
    segment_pins[5] = SEG_F;
    segment_pins[6] = SEG_G;
}

void SevenSegmentDisplay::setup_pins() {
    // Postavljanje pinova za segmente kao izlazni
    for (int i = 0; i < 7; i++) {
        gpio_set_direction((gpio_num_t)segment_pins[i], GPIO_MODE_OUTPUT);
        gpio_set_level((gpio_num_t)segment_pins[i], 0);
    }
    
    // Postavljanje pinova za kontrolu cifara kao izlazni
    gpio_set_direction((gpio_num_t)DIGIT1_COM, GPIO_MODE_OUTPUT);
    gpio_set_direction((gpio_num_t)DIGIT2_COM, GPIO_MODE_OUTPUT);
    
    // Za common cathode: 0 = ON, 1 = OFF
    gpio_set_level((gpio_num_t)DIGIT1_COM, 1);  // 1 = OFF 
    gpio_set_level((gpio_num_t)DIGIT2_COM, 1);  // 1 = OFF 
    
    printf("Pins initialized - Common Cathode configuration\n");
}

void SevenSegmentDisplay::set_segments(uint8_t segments) {
    // Postavljanje segmenata A-G
    for (int i = 0; i < 7; i++) {
        if (segments & (1 << i)) {
            gpio_set_level((gpio_num_t)segment_pins[i], 1);
        } else {
            gpio_set_level((gpio_num_t)segment_pins[i], 0);
        }
    }
}

void SevenSegmentDisplay::display_number_multiplexed(uint8_t number) {
    uint8_t tens = number / 10;
    uint8_t units = number % 10;
    
    // Prikaži desetice (lijeva cifra - DIG2)
    gpio_set_level((gpio_num_t)DIGIT1_COM, 0);  // Isključi desnu cifru
    gpio_set_level((gpio_num_t)DIGIT2_COM, 0);  // Isključi lijevu cifru
    set_segments(digit_to_segments[tens]);
    gpio_set_level((gpio_num_t)DIGIT1_COM, 1);  // Uključi lijevu cifru (0 = ON za common cathode)
    vTaskDelay(3 / portTICK_PERIOD_MS);  // 3ms pauza
    
    // Prikaži jedinice (desna cifra - DIG1)  
    gpio_set_level((gpio_num_t)DIGIT2_COM, 0);  // Isključi lijevu cifru
    gpio_set_level((gpio_num_t)DIGIT1_COM, 0);  // Isključi desnu cifru
    set_segments(digit_to_segments[units]);
    gpio_set_level((gpio_num_t)DIGIT2_COM, 1);  // Uključi desnu cifru (0 = ON za common cathode)
    vTaskDelay(3 / portTICK_PERIOD_MS);  // 3ms pauza
}