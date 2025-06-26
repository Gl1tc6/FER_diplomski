#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
//ANODNO RIJESENJE ------------------------- za katodno bitovi paljena su obrnuti 0 -> 1 a 1 -> 0
// Definicije pinova za segmente A-G (zajednički za oba displaya)
#define SEG_A 2
#define SEG_B 4
#define SEG_C 16
#define SEG_D 17
#define SEG_E 5
#define SEG_F 18
#define SEG_G 19

// Definicije pinova za kontrolu cifara (common cathode)
#define DIGIT1_COM 21  // Desna cifra (jedinice)
#define DIGIT2_COM 22  // Lijeva cifra (desetice)

// Lookup tablica za 7-segmentni displej (common cathode)
// Bit pozicije: G F E D C B A (bit 6 do bit 0)
const uint8_t digit_to_segments[10] = {
    
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

// Pinovi za segmente A-G
int segment_pins[7] = {SEG_A, SEG_B, SEG_C, SEG_D, SEG_E, SEG_F, SEG_G};

void setup_pins() {
    // Postavljanje pinova za segmente kao izlazni
    for (int i = 0; i < 7; i++) {
        gpio_set_direction(segment_pins[i], GPIO_MODE_OUTPUT);
        gpio_set_level(segment_pins[i], 0);
    }
    
    // Postavljanje pinova za kontrolu cifara kao izlazni
    gpio_set_direction(DIGIT1_COM, GPIO_MODE_OUTPUT);
    gpio_set_direction(DIGIT2_COM, GPIO_MODE_OUTPUT);
    
    // Za common cathode: 0 = ON, 1 = OFF
    gpio_set_level(DIGIT1_COM, 1);  // 1 = OFF 
    gpio_set_level(DIGIT2_COM, 1);  // 1 = OFF 
    
    printf("Pins initialized - Common Cathode configuration\n");
}

void set_segments(uint8_t segments) {
    // Postavljanje segmenata A-G
    for (int i = 0; i < 7; i++) {
        if (segments & (1 << i)) {
            gpio_set_level(segment_pins[i], 1);
        } else {
            gpio_set_level(segment_pins[i], 0);
        }
    }
}

void display_number_multiplexed(uint8_t number) {
    uint8_t tens = number / 10;
    uint8_t units = number % 10;
    
    // Prikaži desetice (lijeva cifra - DIG2)
    gpio_set_level(DIGIT1_COM, 0);  // Isključi desnu cifru
    gpio_set_level(DIGIT2_COM, 0);  // Isključi lijevu cifru
    set_segments(digit_to_segments[tens]);
    gpio_set_level(DIGIT1_COM, 1);  // Uključi lijevu cifru (0 = ON za common cathode)
    vTaskDelay(3 / portTICK_PERIOD_MS);  // 3ms pauza
    
    // Prikaži jedinice (desna cifra - DIG1)  
    gpio_set_level(DIGIT2_COM, 0);  // Isključi lijevu cifru
    gpio_set_level(DIGIT1_COM, 0);  // Isključi desnu cifru
    set_segments(digit_to_segments[units]);
    gpio_set_level(DIGIT2_COM, 1);  // Uključi desnu cifru (0 = ON za common cathode)
    vTaskDelay(3 / portTICK_PERIOD_MS);  // 3ms pauza
}

void app_main() {
    setup_pins();
    
    uint8_t counter = 0;
    uint32_t last_time = 0;
    printf("Bokkkkkkkkkk ja sam ----------------------------------------\n");
    printf("ESP32 Dual 7-Segment Counter Started\n");
    printf("Testing display with number 88...\n");
    
    // Test prikaz broja 88 na početak (svi segmenti uključeni)
    for(int i = 0; i < 100; i++) {
        display_number_multiplexed(88);
    }
    
    printf("Starting counter from 00...\n");
    
    while (1) {
        uint32_t current_time = xTaskGetTickCount() * portTICK_PERIOD_MS;
        
        // Kontinuirano multipleksiranje displaya
        display_number_multiplexed(counter);
        
        // Povećaj brojač svakih 500ms
        if (current_time - last_time >= 500) {
            counter++;
            if (counter > 99) {
                counter = 0;
            }
            printf("Displaying: %02d (tens=%d, units=%d)\n", counter, counter/10, counter%10);
            last_time = current_time;
        }
    }
}