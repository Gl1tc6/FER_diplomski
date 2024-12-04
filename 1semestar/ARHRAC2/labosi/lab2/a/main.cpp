#include <iostream>

extern "C" int potprogram_asm(int, int, int);

int potprogram_c(int a, int b, int c){
	return (a+b)*c;
}

int main(){
    	std::cout <<"ASM: " <<potprogram_asm(3,5,6) <<std::endl;
    	std::cout <<"C++: " <<potprogram_c(3,5,6) <<std::endl;
    }