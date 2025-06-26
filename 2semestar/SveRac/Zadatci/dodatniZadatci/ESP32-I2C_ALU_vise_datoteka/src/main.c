#include <stdio.h>
#include "alu_driver.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_adc_cal.h"
#include "esp_timer.h"

#define ALU_PAR1 0x01
#define ALU_PAR2 0x02

#define OPERATION_POWER_A 5
#define OPERATION_POWER_B 6
#define OPERATION_ADD     1
#define OPERATION_SQRT_A  7

void app_main(void) {
    ALU.init();

    float a = 3.0f;
    float b = 4.0f;
    float a2, b2, sum, result;

    for (int i = 1; i < 10; i++){
        a += 1.0;
        b += 1.0;


        ALU.write_reg(ALU_PAR1, a);
        ALU.set_op(OPERATION_POWER_A);
        ALU.read_result(&a2);
        printf("a2 = %f\n", a2);

        ALU.write_reg(ALU_PAR2, b);
        ALU.set_op(OPERATION_POWER_B);
        ALU.read_result(&b2);
        printf("b2 = %f\n", b2);

        ALU.write_reg(ALU_PAR1, a2);
        ALU.write_reg(ALU_PAR2, b2);
        ALU.set_op(OPERATION_ADD);
        ALU.read_result(&sum);
        printf("sum = %f\n", sum);

        ALU.write_reg(ALU_PAR1, sum);
        ALU.set_op(OPERATION_SQRT_A);
        ALU.read_result(&result);

        printf("Hipotenuza za a=%.2f i b=%.2f je: %.2f\n", a, b, result);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
