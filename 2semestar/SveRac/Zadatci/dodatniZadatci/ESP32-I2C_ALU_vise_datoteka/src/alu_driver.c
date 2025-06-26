#include "alu_driver.h"
#include "driver/i2c.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <stdio.h>

#define CONFIG_SCL_GPIO 15
#define CONFIG_SDA_GPIO 5

#define I2C_PORT I2C_NUM_0
#define I2C_FREQ 100000

#define ALU_ADDR 0x22
#define ALU_OPER  0x00

union floatunion_t {
    float f;
    uint8_t b[sizeof(float)];
};

static esp_err_t alu_i2c_init(void) {
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

static esp_err_t alu_write_reg(uint8_t reg, float value) {
    union floatunion_t u;
    u.f = value;
    uint8_t buf[5] = {reg, u.b[0], u.b[1], u.b[2], u.b[3]};
    esp_err_t ret;
    do {
        ret = i2c_master_write_to_device(I2C_PORT, ALU_ADDR, buf, sizeof(buf), pdMS_TO_TICKS(1000));
        printf("Zapeo u pisanju vrijednosti\n");
    } while (ret != ESP_OK);
    printf("Uspješno pisanje vrijednosti\n");
    return ret;
}

static esp_err_t alu_set_op(uint8_t op) {
    uint8_t buf[2] = { ALU_OPER, op };
    esp_err_t ret;
    do {
        ret = i2c_master_write_to_device(I2C_PORT, ALU_ADDR, buf, sizeof(buf), pdMS_TO_TICKS(1000));
        printf("Zapeo u pisanju operacije\n");
    } while (ret != ESP_OK);
    printf("Uspješno pisanje operacije\n");
    return ret;
}

static esp_err_t alu_read_result(float* result) {
    uint8_t buf[4];
    esp_err_t ret;
    do {
        ret = i2c_master_read_from_device(I2C_PORT, ALU_ADDR, buf, sizeof(buf), pdMS_TO_TICKS(1000));
        printf("Zapeo u čitanju rezultata\n");
    } while (ret != ESP_OK);
    printf("Uspješno čitanje rezultata\n");

    union floatunion_t u = { .b = { buf[0], buf[1], buf[2], buf[3] } };
    *result = u.f;
    return ESP_OK;
}

const AluDriver ALU = {
    .init = alu_i2c_init,
    .write_reg = alu_write_reg,
    .set_op = alu_set_op,
    .read_result = alu_read_result,
};
