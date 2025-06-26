#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_log.h"

// Definicije pinova
#define DIG1 14
#define DIG2 32
#define DIG3 33
#define DIG4 25

#define SEGMENT_A 21
#define SEGMENT_B 19
#define SEGMENT_C 18
#define SEGMENT_D 5
#define SEGMENT_E 4
#define SEGMENT_F 2
#define SEGMENT_G 15
#define DP 13
#define CLN 12
#define COM 

#define BUTTON 27

static int hours = 12;
static int minutes = 0;
static int settingMode = 0; // 0=normal, 1=set hour1, 2=set hour2, 3=set min1, 4=set min2
static int64_t lastTime = 0;
static int64_t lastBlink = 0;
static int64_t buttonPressTime = 0;
static bool buttonPressed = false;
static bool colonState = true;

// Segmenti za znamenke 0-9
static const uint8_t digits[10][7] = {
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

static const gpio_num_t segmentPins[7] = {SEGMENT_A, SEGMENT_B, SEGMENT_C, SEGMENT_D, SEGMENT_E, SEGMENT_F, SEGMENT_G};
static const gpio_num_t digitPins[4] = {DIG1, DIG2, DIG3, DIG4};

// prototip funkcija
void display_time(void);
void clock_task(void *pvParameter);
void increment_digit(void);
void handle_button(void);
void init_gpio(void);

void init_gpio(void)
{
    // segmente -> izlaze
    for (int i = 0; i < 7; i++)
    {
        gpio_set_direction(segmentPins[i], GPIO_MODE_OUTPUT);
        gpio_set_level(segmentPins[i], 0);
    }

    // brojevi -> izlazi
    for (int i = 0; i < 4; i++)
    {
        gpio_set_direction(digitPins[i], GPIO_MODE_OUTPUT);
        gpio_set_level(digitPins[i], 1); // Isključi sve znamenke
    }

    gpio_set_direction(DP, GPIO_MODE_OUTPUT);
    gpio_set_direction(CLN, GPIO_MODE_OUTPUT);

    // pull-up
    gpio_set_direction(BUTTON, GPIO_MODE_INPUT);
    gpio_set_pull_mode(BUTTON, GPIO_PULLUP_ONLY);
}

void handle_button(void)
{
    bool currentState = !gpio_get_level(BUTTON);
    int64_t currentTime = esp_timer_get_time() / 1000; // mikro -> mili

    if (currentState && !buttonPressed)
    {
        buttonPressed = true;
        buttonPressTime = currentTime;
    }

    if (!currentState && buttonPressed)
    {
        int64_t pressDuration = currentTime - buttonPressTime;

        if (pressDuration > 1000) // Dugi pritisak
        {
            if (settingMode == 0)
            {
                settingMode = 1; // Počni podešavanje
                printf("Podešavam znamenku %d\n", settingMode);
            }
            else
            {
                settingMode++;
                if (settingMode > 4)
                {
                    settingMode = 0; // Završi podešavanje
                    lastTime = currentTime;
                    printf("Završio podešavanje - %02d:%02d\n", hours, minutes);
                }
                else
                {
                    printf("Podešavam znamenku %d\n", settingMode);
                }
            }
        }
        else if (pressDuration > 50) // Kratki pritisak (debounce)
        {
            if (settingMode > 0)
            {
                increment_digit();
                printf("Vrijeme: %02d:%02d\n", hours, minutes);
            }
        }

        buttonPressed = false;
    }
}

void increment_digit(void)
{
    switch (settingMode)
    {
    case 1: // Prva znamenka sati (0-2)
        int firstDigit = hours / 10;
        int secondDigit = hours % 10;
        if( firstDigit == 1 && secondDigit < 4){
            firstDigit = 2;
        }else if(firstDigit == 2){
            firstDigit = 0;
        }else{
            firstDigit = 1;
        }
        hours = firstDigit * 10 + secondDigit;
        break;
    case 2: // Druga znamenka sati
        {
            int firstDigit = hours / 10;
            int secondDigit = hours % 10;
            if (firstDigit == 2)
            {
                secondDigit = (secondDigit + 1) % 4; // 0-3 za 20-23
            }
            else
            {
                secondDigit = (secondDigit + 1) % 10; // 0-9 za 00-19
            }
            hours = firstDigit * 10 + secondDigit;
        }
        break;
    case 3: // Prva znamenka minuta (0-5)
        {
            int firstDigit = minutes / 10;
            firstDigit = (firstDigit + 1) % 6;
            minutes = firstDigit * 10 + (minutes % 10);
        }
        break;
    case 4: // Druga znamenka minuta (0-9)
        {
            int firstDigit = minutes / 10;
            int secondDigit = minutes % 10;
            secondDigit = (secondDigit + 1) % 10;
            minutes = firstDigit * 10 + secondDigit;
        }
        break;
    }
}

void display_time(void)
{
    int digits_to_show[4];
    digits_to_show[0] = hours / 10;
    digits_to_show[1] = hours % 10;
    digits_to_show[2] = minutes / 10;
    digits_to_show[3] = minutes % 10;

    int64_t currentTime = esp_timer_get_time() / 1000;

    // Multipleksiraj znamenke
    for (int digit = 0; digit < 4; digit++)
    {
        // Isključi sve znamenke
        for (int i = 0; i < 4; i++)
        {
            gpio_set_level(digitPins[i], 0);
        }

        // Postavi segmente za trenutnu znamenku
        // for (int seg = 0; seg < 7; seg++)
        // {
        //     gpio_set_level(segmentPins[seg], digits[digits_to_show[digit]][seg]);
        // }
        gpio_set_level(SEGMENT_A, !digits[digits_to_show[digit]][0]);
        gpio_set_level(SEGMENT_B, !digits[digits_to_show[digit]][1]);
        gpio_set_level(SEGMENT_C, !digits[digits_to_show[digit]][2]);
        gpio_set_level(SEGMENT_D, !digits[digits_to_show[digit]][3]);
        gpio_set_level(SEGMENT_E, !digits[digits_to_show[digit]][4]);
        gpio_set_level(SEGMENT_F, !digits[digits_to_show[digit]][5]);
        gpio_set_level(SEGMENT_G, !digits[digits_to_show[digit]][6]);

        // Upravljaj točkama - SAMO za znamenku koja se trenutno mijenja
        if (settingMode > 0)
        {
            switch (settingMode)
            {
            case 1:
                gpio_set_level(DIG1, 1);
                gpio_set_level(DP, 0);
                break;
            case 2:
                gpio_set_level(DIG2, 1);
                gpio_set_level(DP, 0);
                break;
            case 3:
                gpio_set_level(DIG3, 1);
                gpio_set_level(DP, 0);
                break;
            case 4:
                gpio_set_level(DIG4, 1);
                gpio_set_level(DP, 0);
                break;
            
            default:
                break;
            }
        }else{
            gpio_set_level(DP, 1);
        }
        gpio_set_level(DP, 1);

        // Upravljaj dvotočkom - SAMO u normalnom radu

        if (digit == 1) // Dvotočka se prikazuje nakon druge znamenke
        {
            if (settingMode == 0)
            {
                gpio_set_level(CLN, colonState);
            }
            else
            {
                gpio_set_level(CLN, 1);
            }
            
        }

        // Uključi trenutnu znamenku
        gpio_set_level(digitPins[digit], 1);

        vTaskDelay(3); // pauza za multipleksiranje
    }
}

void clock_task(void *pvParameter)
{
    init_gpio();
    lastTime = esp_timer_get_time() / 1000;
    lastBlink = 0;

    printf("Digitalni sat pokrenut! Dugi pritisak za podešavanje.\n");

    while (1)
    {
        int64_t currentTime = esp_timer_get_time() / 1000;

        handle_button();

        if (settingMode == 0)
        {
            // Normalni rad
            if (currentTime - lastTime >= 60000) // Svaku minutu
            {
                minutes++;
                if (minutes >= 60)
                {
                    minutes = 0;
                    hours++;
                    if (hours >= 24)
                    {
                        hours = 0;
                    }
                }
                lastTime = currentTime;
            }

            // Blink dvotočka svake pola sekunde
            if (currentTime - lastBlink >= 500)
            {
                colonState = !colonState;
                lastBlink = currentTime;
            }
        }

        display_time();
        vTaskDelay(1); // Kratka pauza za multipleksiranje
    }
}

void app_main(void)
{
    printf("Digitalni sat pokrenut!\n");

    // Pokreni sat direktno bez dodatnog taska
    clock_task(NULL);
}