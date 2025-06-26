#pragma once

#include "esp_err.h"
#include "driver/i2c.h"
#include <cstdint>

class AluDriver {
public:
    AluDriver();
    esp_err_t init();
    esp_err_t writeValue(uint8_t reg, float value);
    esp_err_t setOperation(uint8_t op);
    esp_err_t readResult(float& result);

private:
    static constexpr uint8_t ALU_ADDR = 0x22;
    static constexpr gpio_num_t SDA_GPIO = GPIO_NUM_5;
    static constexpr gpio_num_t SCL_GPIO = GPIO_NUM_15;
    static constexpr i2c_port_t I2C_PORT = I2C_NUM_0;
    static constexpr uint32_t I2C_FREQ = 100000;
};
