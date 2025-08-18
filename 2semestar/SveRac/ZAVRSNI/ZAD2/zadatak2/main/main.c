#include <stdio.h>
#include <math.h>
#include "i2c_alu.h"


void app_main(void) {
    i2c_alu_init();

    float a = 10.0f;
    float b = 11.5f;
    float c = 12.7f;

    printf("f(a,b,c) = (b + sqrt(b² - ac))/a\n");
    printf("a=%.2f, b=%.2f, c=%.2f\n", a, b, c);

    float ac = alu_multiply(a, c);
    printf("a*c = %.6f\n", ac);

    float b2 = alu_power(b);
    printf("b² = %.6f\n", b2);

    float b2_minus_ac = alu_subtract(b2, ac);
    printf("b^2 - a*c = %.6f\n", b2_minus_ac);

    float sqrt_val = alu_sqrt(b2_minus_ac);
    printf("sqrt(b^2 - a*c) = %.6f\n", sqrt_val);

    float b_plus_sqrt = alu_add(b, sqrt_val);
    printf("b + sqrt(b^2 - a*c) = %.6f\n", b_plus_sqrt);

    float result = alu_divide(b_plus_sqrt, a);

    printf("\nI2C result: %.6f\n", result);

    float verif = (b + sqrt(b*b - a*c)) / a;
    printf("C lib:  %.6f\n", verif);
}