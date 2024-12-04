#include <iostream>

extern "C" int strojni_potprogram();

int main(){
    	std::cout <<"ASM: " <<strojni_potprogram() <<std::endl;
    }