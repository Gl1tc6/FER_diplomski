#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "SevenSegmentDisplay.h"

SevenSegmentDisplay display;

extern "C" void app_main() {
    display.setup_pins();
    
    uint8_t counter = 0;
    uint32_t last_time = 0;
    
    printf("ESP32 Dual 7-Segment Counter Started\n");
    printf("Testing display with number 88...\n");
    
    // Test prikaz broja 88 na početak (svi segmenti uključeni)
    for(int i = 0; i < 100; i++) {
        display.display_number_multiplexed(88);
    }
    
    printf("Starting counter from 00...\n");
    
    while (1) {
        uint32_t current_time = xTaskGetTickCount() * portTICK_PERIOD_MS;
        
        // Kontinuirano multipleksiranje displaya
        display.display_number_multiplexed(counter);
        
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