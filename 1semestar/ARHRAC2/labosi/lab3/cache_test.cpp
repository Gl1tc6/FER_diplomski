/*
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        48 bits physical, 48 bits virtual
Byte Order:                           Little Endian
CPU(s):                               12
L1d cache:                            192 KiB (6 instances)
L1i cache:                            192 KiB (6 instances)
L2 cache:                             3 MiB (6 instances)
L3 cache:                             8 MiB (2 instances)

Veličine cacheva onzačavaju sumu memorije svih instanci; L1d i L1i su isti cache
*/
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>

constexpr size_t S1 = 32 * 1024;        // L1d: 32 KiB po jezgri (192 KiB ukupno / 6)
constexpr size_t B1 = 64;               // L1 linija (64 bajta)
constexpr size_t S2 = 512 * 1024;       // L2: 512 KiB po jezgri (3 MiB ukupno / 6)
constexpr size_t B2 = 64;               // L2 linija
constexpr size_t S3 = 4 * 1024 * 1024;  // L3: 4 MiB po skupini jezgri (8 MiB ukupno / 2)
constexpr size_t B3 = 64; 

constexpr int REPEAT = 10000;              // Broj ponavljanja mjerenja za pouzdanost

void initialize_memory(std::vector<unsigned char>& memory) {
    std::fill(memory.begin(), memory.end(), 0);
}

void print_bandwidth(double time_spent, size_t data_size) {
    double seconds = time_spent / CLOCKS_PER_SEC;
    double bandwidth = (data_size / (1024.0 * 1024.0)) / seconds; // MB/s
    std::cout << "Vrijeme: " << seconds << " sekundi, Propusnost: " << bandwidth << " MB/s\n";
}

void program_A(std::vector<unsigned char>& memory) {
    for (int r = 0; r < REPEAT; ++r) {
        for (size_t i = 0; i < memory.size(); ++i) {
            memory[i]++;
        }
    }
}

void program_B(std::vector<unsigned char>& memory, size_t stride) {
    for (int r = 0; r < REPEAT; ++r) {
        for (size_t i = 0; i < memory.size(); i += stride) {
            memory[i]++;
        }
    }
}

void program_C(std::vector<unsigned char>& memory, size_t stride) {
    for (int r = 0; r < REPEAT; ++r) {
        for (size_t i = 0; i < memory.size(); i += stride) {
            memory[i]++;
        }
    }
}

void program_D(std::vector<unsigned char>& memory, size_t stride) {
    for (int r = 0; r < REPEAT; ++r) {
        for (size_t i = 0; i < memory.size(); i += stride) {
            memory[i]++;
        }
    }
}

int main() {
    clock_t start, end;
    double time_spent;

    // Potprogram A: veličina spremnika = S1
    std::vector<unsigned char> memory_A(S1, 0);
    initialize_memory(memory_A);
    std::cout << "Potprogram A (L1):\n";
    start = clock();
    program_A(memory_A);
    end = clock();
    time_spent = double(end - start);
    print_bandwidth(time_spent, memory_A.size() * REPEAT);

    // Potprogram B: veličina spremnika = 2 * S1, stride = B1 * delta
    constexpr size_t delta = 8;
    std::vector<unsigned char> memory_B(2 * S1 * delta, 0);
    initialize_memory(memory_B);
    std::cout << "Potprogram B (L1 promašaji, L2):\n";
    start = clock();
    program_B(memory_B, B1 * delta);
    end = clock();
    time_spent = double(end - start);
    print_bandwidth(time_spent, memory_B.size() * REPEAT);

    // Potprogram C: veličina spremnika = 2 * S2, stride = B2 * delta
    std::vector<unsigned char> memory_C(2 * S2 * delta, 0);
    initialize_memory(memory_C);
    std::cout << "Potprogram C (L2 promašaji, L3):\n";
    start = clock();
    program_C(memory_C, B2 * delta);
    end = clock();
    time_spent = double(end - start);
    print_bandwidth(time_spent, memory_C.size() * REPEAT);

    // Potprogram D: veličina spremnika = 2 * S3, stride = B3 * delta
    std::vector<unsigned char> memory_D(2 * S3 * delta, 0);
    initialize_memory(memory_D);
    std::cout << "Potprogram D (L3 promašaji, RAM):\n";
    start = clock();
    program_D(memory_D, B3 * delta);
    end = clock();
    time_spent = double(end - start);
    print_bandwidth(time_spent, memory_D.size() * REPEAT);

    return 0;
}
