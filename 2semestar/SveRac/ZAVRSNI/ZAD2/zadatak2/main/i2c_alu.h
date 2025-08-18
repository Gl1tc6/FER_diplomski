#define I2C_ALU_H

#include "esp_err.h"

#define ALU1_ADDR 0x20  
#define ALU2_ADDR 0x22  
#define ALU3_ADDR 0x24

#define ALU_OPER  0x00
#define ALU_PAR1  0x01
#define ALU_PAR2  0x02

#define I2C_OP_NOP      0x00
#define I2C_OP_1        0x01  // +,*,^
#define I2C_OP_2        0x02  // -,/,SQRT
#define I2C_OP_READ_A   0x64  // READ prvi
#define I2C_OP_READ_B   0x65  // READ drugi

esp_err_t i2c_alu_init(void);
esp_err_t alu_write_reg(uint8_t device_addr, uint8_t reg, float value);
esp_err_t alu_set_op(uint8_t device_addr, uint8_t op);
esp_err_t alu_read_result(uint8_t device_addr, float* result);

float alu_multiply(float a, float b);
float alu_divide(float a, float b);
float alu_add(float a, float b);
float alu_subtract(float a, float b);
float alu_power(float a);
float alu_sqrt(float a);
