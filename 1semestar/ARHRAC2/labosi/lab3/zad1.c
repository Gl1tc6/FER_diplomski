#include <stdio.h>
#include <stdint.h>

// Funkcija za testiranje endianness-a
void test_endianness() {
    // Deklariramo 32-bitni unsigned integer
    uint32_t test_number = 0x12345678;
    
    // Koristimo pokazivač char za pristup pojedinačnim bajtovima
    uint8_t *byte_pointer = (uint8_t*)&test_number;

    printf("Testni broj: 0x%X\n", test_number);
    printf("Redoslijed bajtova: ");
    
    // Ispisujemo bajtove u poretku kako su pohranjeni u memoriji
    for (int i = 0; i < sizeof(uint32_t); i++) {
        printf("%02X ", byte_pointer[i]);
    }
    printf("\n");

    // Određivanje tipa endianness-a
    if (byte_pointer[0] == 0x12) {
        printf("Big Endian arhitektura\n");
    } else if (byte_pointer[0] == 0x78) {
        printf("Little Endian arhitektura\n");
    } else {
        printf("Nepoznata arhitektura\n");
    }
}

// Dodatna funkcija za testiranje na ARM arhitekturi
#ifdef __arm__
void test_arm_endianness() {
    printf("\nTESTIRANJE NA ARM ARHITEKTURI:\n");
    test_endianness();
}
#endif

int main() {
    printf("TESTIRANJE ENDIANNESS-A\n");
    printf("----------------------\n");
    
    // Generički test
    test_endianness();

    // Specifičan test za ARM, ako je moguće
    #ifdef __arm__
    test_arm_endianness();
    #endif

    return 0;
}