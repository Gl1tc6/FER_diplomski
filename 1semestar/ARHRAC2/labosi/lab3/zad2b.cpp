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
L1d cache:                            192 KiB (6 instances)
L1i cache:                            192 KiB (6 instances)
L2 cache:                             3 MiB (6 instances)
L3 cache:                             8 MiB (2 instances)
*/

#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>

#define s1  (192 * 1024 / 6) // 1KiB = 1024B, 192KiB = 192 * 1024B = 196 608B 
#define b1  64
#define s2  (3 * 1024 * 1024 / 6) // 1MiB = 1024KiB = 1024 * 1024 = 1 048 576B
#define b2  64
#define s3  (8 * 1024 * 1024) // L3 memorija je dijeljena sa svim jezgrama procesora pa je nije potrebno djeliti
#define b3  64
#define loops 1000
#define delta 8

using namespace std;

void print_mem_time(double time, size_t size)
{
     double seconds = time / CLOCKS_PER_SEC;
     double bandwidth = (size / (1024.0* 1024)) / seconds;
     cout << "Vrijeme: " << seconds << " sekundi, Propusnost: " << bandwidth << " MB/s\n" << endl;
}

void subprocessA()
{
     vector<unsigned char> memory(s1, 0);
     clock_t start = clock();
     for(int r =0; r < loops; r++)
     {
          for(int i=0; i < memory.size(); i++)
          {
               memory[i]++;
          }
     }
     print_mem_time(double(clock() - start), memory.size() * loops);
}

void subprocessB()
{
     vector<unsigned char> memory(2*s1*delta, 0);
     clock_t start = clock();
     for (int r = 0; r < loops; ++r) 
     {
          for (size_t i = 0; i < memory.size(); i += delta) 
          {
               memory[i]++;
          }
    }
    print_mem_time(double(clock() - start), memory.size() * loops);
}

void subprocessC()
{
     vector<unsigned char> memory(2*s2*delta, 0);
     clock_t start = clock();
     for (int r = 0; r < loops; ++r) 
     {
          for (size_t i = 0; i < memory.size(); i += delta) 
          {
               memory[i]++;
          }
    }
    print_mem_time(double(clock() - start), memory.size() * loops);
}

void subprocessD()
{
     vector<unsigned char> memory(2*s3*delta, 0);
     clock_t start = clock();
     for (int r = 0; r < loops; ++r) 
     {
          for (size_t i = 0; i < memory.size(); i += delta) 
          {
               memory[i]++;
          }
    }
    print_mem_time(double(clock() - start), memory.size() * loops);
}


int main(){
    // std::cout << "s1: " << s1 << "\ts2: " << s2 << "\ts3: " << s3 << std::endl;
    cout << "Program A" << endl;
    subprocessA();

    cout << "Program B" << endl;
    subprocessB();

    cout << "Program C" << endl;
    subprocessC();

    cout << "Program D" << endl;
    subprocessD();

    return 0;
}



