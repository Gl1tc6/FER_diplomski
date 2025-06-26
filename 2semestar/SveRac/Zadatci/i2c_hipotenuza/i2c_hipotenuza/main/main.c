#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "esp_log.h"
#include <stdint.h>


#define CONFIG_SCL_GPIO 15
#define CONFIG_SDA_GPIO 5

#define I2C_PORT I2C_NUM_0
#define I2C_FREQ 100000

#define ALU_ADDR 0x22
#define ALU_OPER  0x00
#define ALU_PAR1  0x01
#define ALU_PAR2  0x02

#define OPERATION_NOP      0
#define OPERATION_ADD      1
#define OPERATION_SUB      2
#define OPERATION_MUL      3
#define OPERATION_DIV      4
#define OPERATION_POWER_A  5
#define OPERATION_POWER_B  6
#define OPERATION_SQRT_A   7
#define OPERATION_SQRT_B   8

union floatunion_t {
    float f;
    uint8_t b[sizeof(float)];
};

// I2C inicijalizacija
esp_err_t i2c_master_init(void) {
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = CONFIG_SDA_GPIO,
        .scl_io_num = CONFIG_SCL_GPIO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_FREQ,
    };
    i2c_param_config(I2C_PORT, &conf);
    return i2c_driver_install(I2C_PORT, conf.mode, 0, 0, 0);
}

// Slanje float vrijednosti u registar
esp_err_t alu_write_reg(uint8_t reg, float value) {
    union floatunion_t u;
    u.f = value;
    esp_err_t ret;
    uint8_t buf[5] = {reg, u.b[0], u.b[1], u.b[2], u.b[3]};
    do{
    ret = i2c_master_write_to_device(I2C_PORT, ALU_ADDR, buf, sizeof(buf), pdMS_TO_TICKS(1000));
    printf("Zapeo u pisanju vrijednosti\n");
    } while (ret != ESP_OK);
    printf("Uspješno pisanje vrijednosti-----------------------------------\n");
    return ret;
}

// Postavljanje operacije
esp_err_t alu_set_op(uint8_t op) {
    uint8_t buf[2] = { ALU_OPER, op };
    esp_err_t ret;
    do{
    ret = i2c_master_write_to_device(I2C_PORT, ALU_ADDR, buf, sizeof(buf), pdMS_TO_TICKS(1000));
    printf("Zapeo u pisanju operacije\n");
    } while (ret != ESP_OK);

    printf("Uspjesno pisanje operacije-----------------------------------\n");
    return ret;
}

// Čitanje rezultata (float)
esp_err_t alu_read_result(float* result) {
    uint8_t buf[4];
    esp_err_t ret;
    do
    {
        ret = i2c_master_read_from_device(I2C_PORT, ALU_ADDR, buf, sizeof(buf), pdMS_TO_TICKS(1000));
        printf("Tu sam u while-u čitanja rezultata\n");
    } while (ret != ESP_OK);
    printf("Tu sam izasao iz while-a rezultata\n");

    union floatunion_t u = { .b = { buf[0], buf[1], buf[2], buf[3] } };
    printf("u.f = %f", u.f);
    *result = u.f;
    return ESP_OK;
}


void app_main(void) {
    i2c_master_init();

    float a = 3.0f;
    float b = 4.0f;
    float a2, b2, sum, result;
    

    // Kvadrat broja a (PAR1 = a, operacija POWER_A)
    alu_write_reg(ALU_PAR1, a);
    alu_set_op(OPERATION_POWER_A);
    alu_read_result(&a2);
    printf("a2 = %f\n", a2);

    // Kvadrat broja b (PAR2 = b, operacija POWER_B)
    alu_write_reg(ALU_PAR2, b);
    alu_set_op(OPERATION_POWER_B);
    alu_read_result(&b2);
    printf("b2 = %f\n", b2);

    // Zbroj kvadrata a² + b²
    alu_write_reg(ALU_PAR1, a2);
    alu_write_reg(ALU_PAR2, b2);
    alu_set_op(OPERATION_ADD);
    alu_read_result(&sum);
    printf("sum = %f\n", sum);

    // Kvadratni korijen iz rezultata
    alu_write_reg(ALU_PAR1, sum);
    alu_set_op(OPERATION_SQRT_A);
    alu_read_result(&result);

    printf("Hipotenuza za a=%.2f i b=%.2f je: %.2f\n", a, b, result);
}