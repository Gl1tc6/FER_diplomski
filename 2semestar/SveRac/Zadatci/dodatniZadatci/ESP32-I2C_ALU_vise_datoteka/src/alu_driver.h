#pragma once

#include "esp_err.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    esp_err_t (*init)(void);
    esp_err_t (*write_reg)(uint8_t reg, float value);
    esp_err_t (*set_op)(uint8_t op);
    esp_err_t (*read_result)(float* result);
} AluDriver;

extern const AluDriver ALU;

#ifdef __cplusplus
}
#endif
