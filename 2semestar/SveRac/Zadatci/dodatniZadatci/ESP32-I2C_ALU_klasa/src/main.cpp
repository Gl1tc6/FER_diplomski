#include <cstdio>
#include "AluDriver.h"

#define PAR1 0x01
#define PAR2 0x02

#define OP_POWER_A 5
#define OP_POWER_B 6
#define OP_ADD     1
#define OP_SQRT_A  7

extern "C" void app_main(void) {
    AluDriver alu;
    alu.init();
    float a = 3.0f, b = 4.0f;
    float a2, b2, sum, result;
    
    for (int i = 0; i < 10; i++){

        

        alu.writeValue(PAR1, a);
        alu.setOperation(OP_POWER_A);
        alu.readResult(a2);

        alu.writeValue(PAR2, b);
        alu.setOperation(OP_POWER_B);
        alu.readResult(b2);

        alu.writeValue(PAR1, a2);
        alu.writeValue(PAR2, b2);
        alu.setOperation(OP_ADD);
        alu.readResult(sum);

        alu.writeValue(PAR1, sum);
        alu.setOperation(OP_SQRT_A);
        alu.readResult(result);

        printf("A = %.2f, B = %.2f, Hipotenuza je: %.2f\n", a, b, result);

        a += 1.0;
        b += 1.0;
        vTaskDelay(100);
    }
}
