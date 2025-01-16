#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
/*
-memory
          description: System Memory
          physical id: 13
          slot: System board or motherboard
          size: 8GiB
        *-bank:0
             description: SODIMM DDR4 Synchronous Unbuffered (Unregistered) 3200 MHz (0.3 ns)
             product: HMA851S6CJR6N-XN
             vendor: Hynix
             physical id: 0
             serial: 00000000
             slot: DIMM 0
             size: 4GiB
             width: 64 bits
             clock: 3200MHz (0.3ns)
        *-bank:1
             description: SODIMM DDR4 Synchronous Unbuffered (Unregistered) 3200 MHz (0.3 ns)
             product: HMA851S6CJR6N-XN
             vendor: Hynix
             physical id: 1
             serial: 00000000
             slot: DIMM 0
             size: 4GiB
             width: 64 bits
             clock: 3200MHz (0.3ns)
     *-cache:0
          description: L1 cache
          physical id: 15
          slot: L1 - Cache
          size: 384KiB
          capacity: 384KiB
          clock: 1GHz (1.0ns)
          capabilities: pipeline-burst internal write-back unified
          configuration: level=1
     *-cache:1
          description: L2 cache
          physical id: 16
          slot: L2 - Cache
          size: 3MiB
          capacity: 3MiB
          clock: 1GHz (1.0ns)
          capabilities: pipeline-burst internal write-back unified
          configuration: level=2
     *-cache:2
          description: L3 cache
          physical id: 17
          slot: L3 - Cache
          size: 8MiB
          capacity: 8MiB
          clock: 1GHz (1.0ns)
          capabilities: pipeline-burst internal write-back unified
          configuration: level=3
     *-cpu
          description: CPU
          product: AMD Ryzen 5 5500U with Radeon Graphics
          vendor: Advanced Micro Devices [AMD]
          physical id: 18
          bus info: cpu@0
          version: 23.104.1
          serial: Unknown
          slot: FP6
          size: 1426MHz
          capacity: 4056MHz
          width: 64 bits
          clock: 100MHz

Veličine cacheva onzačavaju sumu memorije svih instanci; L1d i L1i su isti cache
*/

// Parametri veličina i pomaka za različite razine memorije
#define S1 (32 * 1024)       // Veličina L1 priručne memorije
#define B1 64                // Veličina linije L1 priručne memorije
#define S2 (256 * 1024)      // Veličina L2 priručne memorije
#define B2 64                // Veličina linije L2 priručne memorije
#define S3 (12 * 1024 * 1024)// Veličina L3 priručne memorije
#define B3 64                // Veličina linije L3 priručne memorije
#define DELTA 8              // Pomak za onemogućavanje prefetchinga

// Maksimalno trajanje mjerenja (u sekundama)
#define MAX_TIME 2.0

void flush_cache() {
    const size_t flush_size = 16 * 1024 * 1024; // 16MB
    char* flush_buffer = malloc(flush_size);
    for (size_t i = 0; i < flush_size; i++) {
        flush_buffer[i] = i % 256;
    }
    free(flush_buffer);
}


// Funkcija za mjerenje vremena
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Potprogram A: Sekvencijalni pristup unutar L1
uint64_t program_a() {
    uint8_t* buffer = calloc(S1, sizeof(uint8_t));
    uint64_t sum = 0;
    double start_time = get_time();
    double end_time = start_time;
    int iterations = 0;

    while (end_time - start_time < MAX_TIME) {
        for (int i = 0; i < S1; i++) {
            buffer[i]++;
        }
        
        iterations++;
        end_time = get_time();
    }

    // Računanje sume radi sprečavanja optimizacije
    for (int i = 0; i < S1; i++) {
        sum += buffer[i];
    }

    free(buffer);
    return sum;
}

// Potprogram B: Pristup svakog b1-tog bajta (L1 promašaji)
uint64_t program_b() {
    uint8_t* buffer = calloc(2 * S1 * DELTA, sizeof(uint8_t));
    uint64_t sum = 0;
    double start_time = get_time();
    double end_time = start_time;
    int iterations = 0;

    while (end_time - start_time < MAX_TIME) {
        for (int i = 0; i < 2 * S1 * DELTA; i += B1 * DELTA) {
            buffer[i]++;
        }
        
        iterations++;
        end_time = get_time();
    }

    for (int i = 0; i < 2 * S1 * DELTA; i++) {
        sum += buffer[i];
    }

    free(buffer);
    return sum;
}

// Potprogram C: Pristup svakog b2-tog bajta (L2 promašaji)
uint64_t program_c() {
    uint8_t* buffer = calloc(2 * S2 * DELTA, sizeof(uint8_t));
    uint64_t sum = 0;
    double start_time = get_time();
    double end_time = start_time;
    int iterations = 0;

    while (end_time - start_time < MAX_TIME) {
        for (int i = 0; i < 2 * S2 * DELTA; i += B2 * DELTA) {
            buffer[i]++;
        }
        
        iterations++;
        end_time = get_time();
    }

    for (int i = 0; i < 2 * S2 * DELTA; i++) {
        sum += buffer[i];
    }

    free(buffer);
    return sum;
}

// Potprogram D: Pristup svakog b3-tog bajta (L3 i RAM promašaji)
uint64_t program_d() {
    uint8_t* buffer = calloc(2 * S3 * DELTA, sizeof(uint8_t));
    uint64_t sum = 0;
    double start_time = get_time();
    double end_time = start_time;
    int iterations = 0;

    while (end_time - start_time < MAX_TIME) {
        for (int i = 0; i < 2 * S3 * DELTA; i += B3 * DELTA) {
            buffer[i]++;
        }
        
        iterations++;
        end_time = get_time();
    }

    for (int i = 0; i < 2 * S3 * DELTA; i++) {
        sum += buffer[i];
    }

    free(buffer);
    return sum;
}

// Funkcija za mjerenje propusnosti i vremena pristupa
void measure_performance(const char* name, uint64_t (*program)(), size_t buffer_size) {
    flush_cache();
    double start_time = get_time();
    uint64_t result = program();
    double end_time = get_time();
    
    double total_time = end_time - start_time;
    double bytes_accessed = (double)buffer_size * sizeof(uint8_t);
    double throughput = (bytes_accessed / (1024.0 * 1024.0)) / total_time; // MB/s
    double avg_access_time = (total_time * 1e9) / bytes_accessed; // nanosekunde
    
    printf("%s:\n", name);
    printf("  Suma: %lu\n", result);
    printf("  Ukupno vrijeme: %.3f s\n", total_time);
    printf("  Propusnost: %.2f MB/s\n", throughput);
    printf("  Prosječno vrijeme pristupa: %.3f ns\n\n", avg_access_time);
}

int main() {
    printf("Benchmark performansi memorijske hijerarhije\n\n");
    
    measure_performance("Program A (L1)", program_a, S1);
    measure_performance("Program B (L1 promašaji)", program_b, 2 * S1 * DELTA);
    measure_performance("Program C (L2 promašaji)", program_c, 2 * S2 * DELTA);
    measure_performance("Program D (L3/RAM promašaji)", program_d, 2 * S3 * DELTA);

    return 0;
}