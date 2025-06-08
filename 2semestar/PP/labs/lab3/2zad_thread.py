import threading
import time
import math
import sys
import argparse
from typing import List
import os

class PiCalculator:
    def __init__(self, N: int, M: int, num_threads: int = None):
        self.N = N
        self.M = M
        self.num_threads = num_threads or os.cpu_count()
        self.partial_sums = [0.0] * self.num_threads
        self.lock = threading.Lock()
        self.global_sum = 0.0
        
    def compute_partial_pi(self, thread_id: int, start_index: int, num_elements: int):
        local_sum = 0.0
        
        for i in range(num_elements):
            n = start_index + i
            if n >= self.N:
                break
            term = (-1)**n / (2*n + 1)
            local_sum += term
        
        # Spremi lokalni rezultat
        self.partial_sums[thread_id] = local_sum
        
        # Thread-safe dodavanje u globalni zbroj
        with self.lock:
            self.global_sum += local_sum
    
    def calculate_parallel(self) -> tuple:
        threads = []
        thread_info = []
        
        start_time = time.time()
        
        # ISPRAVKA: Pravilno dijeljenje posla između dretvi
        # Svaka dretva obrađuje N/num_threads elemenata (plus ostatak za prve dretve)
        elements_per_thread = self.N // self.num_threads
        remaining_elements = self.N % self.num_threads
        
        current_start = 0
        
        # Stvaranje i pokretanje dretvi
        for i in range(self.num_threads):
            # Raspodjeli ostatak među prve dretve
            num_elements = elements_per_thread + (1 if i < remaining_elements else 0)
            
            if num_elements <= 0:
                break
                
            thread_info.append({
                'thread_id': i,
                'start_index': current_start,
                'num_elements': num_elements,
                'end_index': current_start + num_elements - 1
            })
            
            thread = threading.Thread(
                target=self.compute_partial_pi,
                args=(i, current_start, num_elements)
            )
            threads.append(thread)
            thread.start()
            
            current_start += num_elements
        
        # Čekanje završetka svih dretvi
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        pi_value = 4.0 * self.global_sum
        
        return pi_value, execution_time, thread_info

def compute_pi_sequential(N: int) -> float:
    sum_val = 0.0
    for i in range(N):
        term = (-1)**i / (2*i + 1)
        sum_val += term
    return 4.0 * sum_val

def main():
    parser = argparse.ArgumentParser(description='Računanje Pi pomoću threading')
    parser.add_argument('N', type=int, help='Ukupan broj elemenata reda')
    parser.add_argument('M', type=int, help='Broj elemenata po dretvi (informativno)')
    parser.add_argument('-t', '--threads', type=int, 
                       help='Broj dretvi (zadano: broj CPU-a)')
    
    args = parser.parse_args()
    
    N, M = args.N, args.M
    num_threads = args.threads or os.cpu_count()
    
    print(f"Računanje Pi s N={N}, M={M}, dretve={num_threads}")
    print(f"Broj dostupnih CPU-a: {os.cpu_count()}")
    print("-" * 50)
    
    print("Sekvencijalno...")
    start_seq = time.time()
    pi_sequential = compute_pi_sequential(N)
    end_seq = time.time()
    time_sequential = end_seq - start_seq
    
    print(f"Rezultat: Pi = {pi_sequential:.10f} ({time_sequential:.4f} s)")
    
    print("\nParalelno...")
    calculator = PiCalculator(N, M, num_threads)
    pi_parallel, time_parallel, thread_info = calculator.calculate_parallel()
    
    speedup = time_sequential / time_parallel if time_parallel > 0 else 0
    efficiency = (speedup / num_threads) * 100
    error_percentage = abs(pi_parallel - math.pi) / math.pi * 100
    
    print(f"Rezultat:  Pi = {pi_parallel:.10f} ({time_parallel:.4f} s)")
    print(f"Točna vrijednost:    Pi = {math.pi:.10f}")
    print(f"Ubrzanje:            {speedup:.2f}x")
    print(f"Efikasnost:          {efficiency:.2f}%")
    print(f"Greška:              {error_percentage:.6f}%")
    

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python3 2zad_thread.py <N> <M> [-t THREADS]")
        sys.exit(1)
    
    main()