#include <DS1307.hpp>
#include <time.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <driver/gpio.h>

extern "C" void app_main()
{
    DS1307 rtc = DS1307(I2C_NUM_0, GPIO_NUM_21, GPIO_NUM_22, 5);
    printf("Construct success!\n");
    rtc.init();
    printf("INIT success!\n");

    printf("Čitam vrijeme...\n");
    rtc.read_time();
    printf("Provjera nakon %d sekundi...\n", rtc.get_timeout());
    vTaskDelay(pdMS_TO_TICKS(1000*rtc.get_timeout()));
    rtc.read_time();

    vTaskDelay(pdMS_TO_TICKS(1000));
    int h=0, min=0, s=0;
    printf("Postavi %02d h : %02d min : %02d s\n", h, min, s);
    rtc.set_time(h, min, s);
    rtc.read_time();

    vTaskDelay(pdMS_TO_TICKS(1000));
    // 0x00-06 time; 0x07-3F sram
    uint8_t reg=0x3D;
    uint8_t data=0xDA;
    printf("Pišem 0x%02X na adresu 0x%02X\n", data, reg);
    rtc.write_reg(reg, data);
    vTaskDelay(pdMS_TO_TICKS(1000));

    printf("provjera:\n");
    rtc.read_reg(reg);

}