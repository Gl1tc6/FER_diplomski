#include "AluDriver.h"
#include "driver/i2c.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <cstdio>

union FloatUnion {
    float f;
    uint8_t b[sizeof(float)];
};

AluDriver::AluDriver() {}

esp_err_t AluDriver::init() {
    i2c_config_t conf = {};
    conf.mode = I2C_MODE_MASTER;
    conf.sda_io_num = SDA_GPIO;
    conf.scl_io_num = SCL_GPIO;
    conf.sda_pullup_en = GPIO_PULLUP_ENABLE;
    conf.scl_pullup_en = GPIO_PULLUP_ENABLE;
    conf.master.clk_speed = I2C_FREQ;

    i2c_param_config(I2C_PORT, &conf);
    return i2c_driver_install(I2C_PORT, conf.mode, 0, 0, 0);
}


esp_err_t AluDriver::writeValue(uint8_t reg, float value) {
    FloatUnion u;
    u.f = value;
    uint8_t buf[5] = {reg, u.b[0], u.b[1], u.b[2], u.b[3]};
    esp_err_t ret;
    do {
        ret = i2c_master_write_to_device(I2C_PORT, ALU_ADDR, buf, sizeof(buf), pdMS_TO_TICKS(1000));
        printf("Pokušaj pisanja vrijednosti...\n");
    } while (ret != ESP_OK);
    return ret;
}

esp_err_t AluDriver::setOperation(uint8_t op) {
    uint8_t buf[2] = {0x00, op};
    esp_err_t ret;
    do {
        ret = i2c_master_write_to_device(I2C_PORT, ALU_ADDR, buf, sizeof(buf), pdMS_TO_TICKS(1000));
        printf("Pokušaj postavljanja operacije...\n");
    } while (ret != ESP_OK);
    return ret;
}

esp_err_t AluDriver::readResult(float& result) {
    uint8_t buf[4];
    esp_err_t ret;
    do {
        ret = i2c_master_read_from_device(I2C_PORT, ALU_ADDR, buf, sizeof(buf), pdMS_TO_TICKS(1000));
        printf("Pokušaj čitanja rezultata...\n");
    } while (ret != ESP_OK);

    FloatUnion u = {.b = {buf[0], buf[1], buf[2], buf[3]}};
    result = u.f;
    return ESP_OK;
}
