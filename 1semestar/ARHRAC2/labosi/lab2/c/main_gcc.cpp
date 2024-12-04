#include <iostream>
using namespace std;

extern "C" int zbroj_asm(int);

int zbroj_c(int a){
	int sum = 0;
    for (int i = 1; i < a; i++){
        sum += i;
    }
    return sum;
}

int main(){
    for(int i=1; i<11; i++){
        cout << "N = " << i*2 << endl; 
        cout <<"ASM: " <<zbroj_asm(i*2) <<endl;
    	cout <<"C++: " <<zbroj_c(i*2) <<endl;
        cout << "-----------------------------" << endl;
    }
    cout << "N = " << -3 << endl; 
    cout <<"ASM: " <<zbroj_asm(-3) <<endl;
    cout <<"C++: " <<zbroj_c(-3) <<endl;


}