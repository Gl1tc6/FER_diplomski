#include "i2c_alu.h"
#include "driver/i2c.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <stdint.h>

#define CONFIG_SCL_GPIO_12 15
#define CONFIG_SDA_GPIO_12 5
#define CONFIG_SCL_GPIO_3 26
#define CONFIG_SDA_GPIO_3 25
#define I2C_PORT_12 I2C_NUM_0    // ALU1 i ALU2
#define I2C_PORT_3 I2C_NUM_1     // ALU3
#define I2C_FREQ 100000

union floatunion_t {
    float f;
    uint8_t b[sizeof(float)];
};

static i2c_port_t get_i2c_port(uint8_t device_addr) {
    if (device_addr == ALU3_ADDR) {
        return I2C_PORT_3; 
    } else {
        return I2C_PORT_12;
    }
}

esp_err_t i2c_alu_init(void) {
    esp_err_t ret;
    // init i2c za alu1 i alu2
    i2c_config_t conf1 = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = CONFIG_SDA_GPIO_12,
        .scl_io_num = CONFIG_SCL_GPIO_12,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_FREQ,
    };
    i2c_param_config(I2C_PORT_12, &conf1);
    ret = i2c_driver_install(I2C_PORT_12, conf1.mode, 0, 0, 0);
    if (ret != ESP_OK) return ret;
    
    // init i2c za alu3
    i2c_config_t conf2 = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = CONFIG_SDA_GPIO_3,
        .scl_io_num = CONFIG_SCL_GPIO_3,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_FREQ,
    };
    i2c_param_config(I2C_PORT_3, &conf2);
    ret = i2c_driver_install(I2C_PORT_3, conf2.mode, 0, 0, 0);
    
    return ret;
}

esp_err_t alu_write_reg(uint8_t device_addr, uint8_t reg, float value) {
    union floatunion_t u;
    u.f = value;
    esp_err_t ret;
    uint8_t buf[5] = {reg, u.b[0], u.b[1], u.b[2], u.b[3]};
    i2c_port_t port = get_i2c_port(device_addr);
    do {
        ret = i2c_master_write_to_device(port, device_addr, buf, sizeof(buf), 1000);
    } while (ret != ESP_OK);
    return ret;
}

esp_err_t alu_set_op(uint8_t device_addr, uint8_t op) {
    uint8_t buf[2] = { ALU_OPER, op };
    esp_err_t ret;
    i2c_port_t port = get_i2c_port(device_addr);
    do {
        ret = i2c_master_write_to_device(port, device_addr, buf, sizeof(buf), 1000);
    } while (ret != ESP_OK);
    return ret;
}

esp_err_t alu_read_result(uint8_t device_addr, float* result) {
    uint8_t buf[4];
    esp_err_t ret;
    i2c_port_t port = get_i2c_port(device_addr);
    do {
        ret = i2c_master_read_from_device(port, device_addr, buf, sizeof(buf), 1000);
    } while (ret != ESP_OK);
    
    union floatunion_t u = { .b = { buf[0], buf[1], buf[2], buf[3] } };
    *result = u.f;
    return ESP_OK;
}

float alu_multiply(float a, float b) {
    float result;
    alu_write_reg(ALU2_ADDR, ALU_PAR1, a);
    vTaskDelay(10);
    alu_write_reg(ALU2_ADDR, ALU_PAR2, b);
    vTaskDelay(10);
    alu_set_op(ALU2_ADDR, I2C_OP_1);  // MUL
    vTaskDelay(10);
    alu_read_result(ALU2_ADDR, &result);
    vTaskDelay(10);
    return result;
}

float alu_divide(float a, float b) {
    float result;
    alu_write_reg(ALU2_ADDR, ALU_PAR1, a);
    vTaskDelay(10);
    alu_write_reg(ALU2_ADDR, ALU_PAR2, b);
    vTaskDelay(10);
    alu_set_op(ALU2_ADDR, I2C_OP_2);  // DIV
    vTaskDelay(10);
    alu_read_result(ALU2_ADDR, &result);
    vTaskDelay(10);
    return result;
}

float alu_add(float a, float b) {
    float result;
    alu_write_reg(ALU1_ADDR, ALU_PAR1, a);
    vTaskDelay(10);
    alu_write_reg(ALU1_ADDR, ALU_PAR2, b);
    vTaskDelay(10);
    alu_set_op(ALU1_ADDR, I2C_OP_1);  // ADD
    vTaskDelay(10);
    alu_read_result(ALU1_ADDR, &result);
    vTaskDelay(10);
    return result;
}

float alu_subtract(float a, float b) {
    float result;
    alu_write_reg(ALU1_ADDR, ALU_PAR1, a);
    vTaskDelay(10);
    alu_write_reg(ALU1_ADDR, ALU_PAR2, b);
    vTaskDelay(10);
    alu_set_op(ALU1_ADDR, I2C_OP_2);  // SUB
    vTaskDelay(10);
    alu_read_result(ALU1_ADDR, &result);
    vTaskDelay(10);
    return result;
}

float alu_power(float a) {
    float result;
    alu_write_reg(ALU3_ADDR, ALU_PAR1, a);
    vTaskDelay(10);
    alu_set_op(ALU3_ADDR, I2C_OP_1);  // POWER
    vTaskDelay(10);
    alu_read_result(ALU3_ADDR, &result);
    vTaskDelay(10);
    return result;
}

float alu_sqrt(float a) {
    float result;
    alu_write_reg(ALU3_ADDR, ALU_PAR1, a);
    vTaskDelay(10);
    alu_set_op(ALU3_ADDR, I2C_OP_2);  // SQRT
    vTaskDelay(10);
    alu_read_result(ALU3_ADDR, &result);
    vTaskDelay(10);
    return result;
}
