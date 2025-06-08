import numpy as np
import time
import math
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

def cpu_prime_check(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    
    limit = int(math.isqrt(n)) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            return False
    return True

def count_primes_range(args):
    start, end, array = args
    return sum(1 for i in range(start, end) if cpu_prime_check(array[i]))

# Shared counter for atomic simulation
shared_result = None
result_lock = threading.Lock()

def count_primes_atomic_thread(args):
    global shared_result
    start, end, array, use_atomic = args
    local_count = sum(1 for i in range(start, end) if cpu_prime_check(array[i]))
    
    if use_atomic:
        with result_lock:
            shared_result.value += local_count
    else:
        # Non-atomic (unsafe)
        old_val = shared_result.value
        time.sleep(0.001)  # Simulate processing delay to increase race condition chance
        shared_result.value = old_val + local_count

def make_ranges(N, num_workers):
    chunk_size = N // num_workers
    return [(i * chunk_size, 
             (i + 1) * chunk_size if i < num_workers - 1 else N) 
            for i in range(num_workers)]

def test_parallel_method(method_name, N, host_array, test_configs):
    print(f"\n=== {method_name} ===")
    print("G (global) | L (local) | Time (s) | Speedup | Efficiency")
    print("---------------------------------------------------------")
    
    single_worker_time = None
    
    for global_size, local_size in test_configs:
        try:
            start_time = time.time()
            ranges = make_ranges(N, global_size)
            
            if method_name == "Multiprocessing":
                args = [(start, end, host_array) for start, end in ranges]
                with ProcessPoolExecutor(max_workers=global_size) as executor:
                    results = list(executor.map(count_primes_range, args))
                result = sum(results)
            
            elif method_name == "Threading":
                args = [(start, end, host_array) for start, end in ranges]
                with ThreadPoolExecutor(max_workers=global_size) as executor:
                    results = list(executor.map(count_primes_range, args))
                result = sum(results)
            
            elapsed = (time.time() - start_time)
            
            # Use first measurement as baseline for this method
            if single_worker_time is None:
                single_worker_time = elapsed
                speedup = 1.0
                efficiency = 100.0
            else:
                speedup = single_worker_time / elapsed
                efficiency = (speedup / global_size) * 100
            
            local_str = str(local_size) if local_size else "None"
            print(f"    {global_size:2d}     |   {local_str:4s}   | {elapsed:>8.2f} | {speedup:>6.2f}x | {efficiency:>6.1f}%")
            
        except Exception:
            local_str = str(local_size) if local_size else "None"
            print(f"    {global_size:2d}     |   {local_str:4s}   | {'ERROR':>8} | {'--':>7} | {'--':>6}")

def test_atomic_operations(N, host_array, reference_result):
    global shared_result
    print(f"\n=== Atomic vs Non-Atomic (Threading) ===")
    print("Method     | Workers | Result | Consistency")
    print("-----------------------------------------")
    
    for num_workers in [2, 4, 8]:
        ranges = make_ranges(N, num_workers)
        
        # Test atomic (safe)
        try:
            shared_result = mp.Value('i', 0)
            args = [(start, end, host_array, True) for start, end in ranges]
            
            threads = []
            for arg in args:
                t = threading.Thread(target=count_primes_atomic_thread, args=(arg,))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            atomic_result = shared_result.value
            consistency = "Safe" if atomic_result == reference_result else "Error"
            print(f"Atomic     |    {num_workers}    |   {atomic_result}  |   {consistency}")
            
        except Exception:
            print(f"Atomic     |    {num_workers}    |   --   |    Error")
        
        # Test non-atomic (unsafe) - run multiple times to show inconsistency
        try:
            results = []
            for trial in range(3):
                shared_result = mp.Value('i', 0)
                args = [(start, end, host_array, False) for start, end in ranges]
                
                threads = []
                for arg in args:
                    t = threading.Thread(target=count_primes_atomic_thread, args=(arg,))
                    threads.append(t)
                    t.start()
                
                for t in threads:
                    t.join()
                
                results.append(shared_result.value)
            
            unique_results = len(set(results))
            consistency = "Safe" if unique_results == 1 and results[0] == reference_result else "Unsafe"
            avg_result = sum(results) / len(results)
            
            print(f"Non-atomic |    {num_workers}    | {avg_result:>6.1f}  |  {consistency}")
            
        except Exception:
            print(f"Non-atomic |    {num_workers}    |   --   |    Error")

def main():
    N = 2**18
    host_array = np.arange(N, dtype=np.int32)
    
    print(f"Prime Counter Analysis (N={N})")
    print("=" * 50)
    
    # Reference calculation
    start_time = time.time()
    reference_result = sum(1 for i in range(N) if cpu_prime_check(i))
    baseline_time = (time.time() - start_time) * 1000
    
    print(f"Reference: {reference_result} primes, {baseline_time:.2f} ms")
    
    # Test configurations: (global_workers, local_group_size)
    # For threading, local_group_size simulates workgroup concept
    test_configs = [
        (1, None),   # Sequential
        (2, 1),      # 2 workers, no grouping
        (4, 2),      # 4 workers, groups of 2
        (8, 4),      # 8 workers, groups of 4
        (16, 8),     # 16 workers, groups of 8
    ]
    
    # Test multiprocessing
    test_parallel_method("Multiprocessing", N, host_array, test_configs)
    
    # Test threading  
    test_parallel_method("Threading", N, host_array, test_configs)
    
    # Test atomic operations
    test_atomic_operations(N, host_array, reference_result)
    
    print(f"\nSystem: {mp.cpu_count()} CPU cores")

if __name__ == "__main__":
    main()