#ifndef SEVEN_SEGMENT_DISPLAY_H
#define SEVEN_SEGMENT_DISPLAY_H

#include <stdint.h>

class SevenSegmentDisplay {
private:
    // Definicije pinova za segmente A-G
    static const int SEG_A = 2;
    static const int SEG_B = 4;
    static const int SEG_C = 16;
    static const int SEG_D = 17;
    static const int SEG_E = 5;
    static const int SEG_F = 18;
    static const int SEG_G = 19;

    // Definicije pinova za kontrolu cifara (common cathode)
    static const int DIGIT1_COM = 21;  // Desna cifra (jedinice)
    static const int DIGIT2_COM = 22;  // Lijeva cifra (desetice)

    // Lookup tablica za 7-segmentni displej (common cathode)
    static const uint8_t digit_to_segments[10];
    
    // Pinovi za segmente A-G
    int segment_pins[7];

    void set_segments(uint8_t segments);

public:
    SevenSegmentDisplay();
    void setup_pins();
    void display_number_multiplexed(uint8_t number);
};
#endif