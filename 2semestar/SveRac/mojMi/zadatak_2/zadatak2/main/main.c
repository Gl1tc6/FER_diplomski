#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/adc.h"
#include "driver/gpio.h"
#include "esp_timer.h"

// Definicije pinova
#define DIG_1 12
#define DIG_2 14
#define SEGMENT_A 19
#define SEGMENT_B 21
#define SEGMENT_C 32
#define SEGMENT_D 33
#define SEGMENT_E 25
#define SEGMENT_F 27
#define SEGMENT_G 13
#define SIGNAL_PIN 26

// Definicije segmenata za brojeve 0-9
const int segments[10][7] = {
    {1, 1, 1, 1, 1, 1, 0}, // 0
    {0, 1, 1, 0, 0, 0, 0}, // 1
    {1, 1, 0, 1, 1, 0, 1}, // 2
    {1, 1, 1, 1, 0, 0, 1}, // 3
    {0, 1, 1, 0, 0, 1, 1}, // 4
    {1, 0, 1, 1, 0, 1, 1}, // 5
    {1, 0, 1, 1, 1, 1, 1}, // 6
    {1, 1, 1, 0, 0, 0, 0}, // 7
    {1, 1, 1, 1, 1, 1, 1}, // 8
    {1, 1, 1, 1, 0, 1, 1}  // 9
};

// Globalne varijable za mjerenje frekvencije
volatile int64_t last_cross_time = 0;
volatile int period_count = 0;
volatile float frequency = 50.0; // početna vrijednost
const int NUM_PERIODS_TO_MEASURE = 5; // broj perioda za precizniji izračun

void setup_gpio() {
    // Konfiguracija pinova kao izlazne
    gpio_set_direction(DIG_1, GPIO_MODE_OUTPUT);
    gpio_set_direction(DIG_2, GPIO_MODE_OUTPUT);
    gpio_set_direction(SEGMENT_A, GPIO_MODE_OUTPUT);
    gpio_set_direction(SEGMENT_B, GPIO_MODE_OUTPUT);
    gpio_set_direction(SEGMENT_C, GPIO_MODE_OUTPUT);
    gpio_set_direction(SEGMENT_D, GPIO_MODE_OUTPUT);
    gpio_set_direction(SEGMENT_E, GPIO_MODE_OUTPUT);
    gpio_set_direction(SEGMENT_F, GPIO_MODE_OUTPUT);
    gpio_set_direction(SEGMENT_G, GPIO_MODE_OUTPUT);
    
    // Isključi obje znamenke
    gpio_set_level(DIG_1, 0);
    gpio_set_level(DIG_2, 0);
    
    // Postavi ADC kanal za mjerenje signala
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC1_CHANNEL_0, ADC_ATTEN_DB_11); // Pretpostavljam da SIGNAL_PIN odgovara ADC1_CHANNEL_0
}

// Prikazuje pojedinu znamenku na 7-segmentnom displeju
void display_digit(int digit, int is_first_digit) {
    // Isključi obje znamenke prije postavke segmenata
    gpio_set_level(DIG_1, 0);
    gpio_set_level(DIG_2, 0);
    
    // Postavi segmente prema znamenki
    gpio_set_level(SEGMENT_A, segments[digit][0]);
    gpio_set_level(SEGMENT_B, segments[digit][1]);
    gpio_set_level(SEGMENT_C, segments[digit][2]);
    gpio_set_level(SEGMENT_D, segments[digit][3]);
    gpio_set_level(SEGMENT_E, segments[digit][4]);
    gpio_set_level(SEGMENT_F, segments[digit][5]);
    gpio_set_level(SEGMENT_G, segments[digit][6]);
    
    // Upali odgovarajuću znamenku
    if (is_first_digit) {
        gpio_set_level(DIG_1, 1);
    } else {
        gpio_set_level(DIG_2, 1);
    }
}

// Zadatak za mjerenje frekvencije
void measure_frequency_task(void *pvParameters) {
    int last_adc_value = 0;
    int cross_detected = 0;
    int64_t current_time;
    int64_t total_time = 0;
    int periods_measured = 0;
    
    while(1) {
        // Čitaj ADC vrijednost (12-bitni ADC daje vrijednosti od 0-4095)
        int adc_value = adc1_get_raw(ADC1_CHANNEL_0);
        int threshold = 2048; // Približno 2.5V na 12-bitnom ADC-u
        
        // Detektiraj prolazak kroz nulu (2.5V) u pozitivnom smjeru
        if (last_adc_value < threshold && adc_value >= threshold) {
            current_time = esp_timer_get_time(); // vrijeme u mikrosekundama
            
            if (last_cross_time > 0) {
                // Dodaj vrijeme perioda
                total_time += (current_time - last_cross_time);
                periods_measured++;
                
                // Nakon mjerenja određenog broja perioda, izračunaj frekvenciju
                if (periods_measured >= NUM_PERIODS_TO_MEASURE) {
                    // Izračunaj frekvenciju (f = 1/T)
                    frequency = 1000000.0 * periods_measured / total_time;
                    printf("Izmjerena frekvencija: %.2f Hz\n", frequency);
                    
                    // Resetiraj brojače za sljedeće mjerenje
                    total_time = 0;
                    periods_measured = 0;
                }
            }
            last_cross_time = current_time;
        }
        
        last_adc_value = adc_value;
        vTaskDelay(1 / portTICK_PERIOD_MS); // Mala pauza za stabilizaciju
    }
}

// Zadatak za prikaz na 7-segmentnom displeju
void display_task(void *pvParameters) {
    while(1) {
        // Zaokruži frekvenciju na najbližu cjelobrojnu vrijednost
        int freq_value = (int)(frequency + 0.5);
        
        // Ograniči na 99 za dvoznamenkasti displej
        if (freq_value > 99) freq_value = 99;
        
        // Izračunaj desetice i jedinice
        int tens = freq_value / 10;
        int ones = freq_value % 10;
        
        // Prikaži desetice
        display_digit(tens, 1);
        vTaskDelay(10 / portTICK_PERIOD_MS); // 10ms (brže od 100ms za bolji vizualni efekt)
        
        // Prikaži jedinice
        display_digit(ones, 0);
        vTaskDelay(10 / portTICK_PERIOD_MS);
    }
}

void app_main() {
    // Inicijalizacija GPIO-a
    setup_gpio();
    
    // Pokreni zadatke
    xTaskCreate(measure_frequency_task, "measure_freq", 2048, NULL, 5, NULL);
    xTaskCreate(display_task, "display", 2048, NULL, 4, NULL);
}