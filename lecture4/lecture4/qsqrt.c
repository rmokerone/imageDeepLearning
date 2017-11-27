#include <stdio.h>

float Q_rsqrt(float number)
{
    long i;
    float x2, y;
    const float threehalfs = 1.5F;
    x2 = number * 0.5F;
    y = number;
    i = * ( long * ) &y; 
    i = 0x5f3759df - ( i >> 1 ); 
    y = * ( float * ) &i;
    y = y * ( threehalfs - ( x2 * y * y ) ); 
    return y;
}

int main(void){
    printf("hello zxx\n");
    printf("1/sqrt(3) = %f\n", Q_rsqrt(300));
}
