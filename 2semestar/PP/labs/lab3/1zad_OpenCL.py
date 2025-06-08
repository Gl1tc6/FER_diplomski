import numpy as np
import pyopencl as cl
import time
import math
import warnings
warnings.filterwarnings("ignore", message="Non-empty compiler output")

def isPrime(n):
    if n <= 1:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i <= math.sqrt(n):
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def main():
    N = 2**18  # 1048576 elemenata
    print(f"Number of elements: {N}")
    
    host_array = np.arange(N, dtype=np.int32)
    
    # init
    platforms = cl.get_platforms()
    if not platforms:
        print("No OpenCL platforms found!")
        return
        
    devices = []
    for platform in platforms:
        try:
            devices.extend(platform.get_devices())
        except:
            pass
            
    if not devices:
        print("No OpenCL devices found!")
        return
        
    device = devices[0]
    print(f"Using device: {device.name}")
    
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    kernel_code = """
    // code from: https://www.geeksforgeeks.org/check-for-prime-number/
    bool is_prime(int n) {
        if (n <= 1)
            return false;

        if (n == 2 || n == 3)
            return true;

        if (n % 2 == 0 || n % 3 == 0)
            return false;
        
        for (int i = 5; i*i <= n; i = i + 6)
            if (n % i == 0 || n % (i + 2) == 0)
                return false;

        return true;
    }

    __kernel void count_primes(
        __global const int* array,
        __global int* result,
        const int n,
        const int use_atomic
    ) {
        int gid = get_global_id(0);
        int total_threads = get_global_size(0);
        int work_per_thread = (n + total_threads - 1) / total_threads;
        int start = gid * work_per_thread;
        int end = min(start + work_per_thread, n);
        
        int local_count = 0;
        for (int i = start; i < end; i++) {
            if (is_prime(array[i])) {
                local_count++;
            }
        }
        
        if (use_atomic) {
            // Safe atomic operation
            atomic_add(result, local_count);
        } else {
            // Non-atomic operation (unsafe but for testing)
            if (local_count > 0) {
                *result += local_count;
            }
        }
    }
    """
    
    try:
        program = cl.Program(ctx, kernel_code).build(options=["-cl-std=CL1.1"])
    except cl.LogicError as e:
        print(f"Build error: {e}")
        return
        
    # Prepare OpenCL buffers
    mf = cl.mem_flags
    d_array = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_array)
    d_result = cl.Buffer(ctx, mf.READ_WRITE, size=4)
    
    # Test configurations (smaller set)
    test_cases = [
        (1, 1),    # Sequential
        (2, 1),    # 2 threads
        (4, 2),    # 2 groups
        (8, 4),    # 2 groups
        (16, 8),   # 2 groups
        (32, 8),  # 2 groups
        (64, 8),
        (128, 16),
        (256, 8),
        (256, 16),
    ]
    cpu_arr = [x for x in range(1, N+1)]
    start_time = time.time()
    cpu_result = 0
    
    for el in cpu_arr:
        if isPrime(el):
            cpu_result += 1
    elapsed = (time.time() - start_time)
    print(f"CPU ref cnt: {cpu_result}")
    print(f"CPU elapsed: {elapsed:.4f} s")
    
    print("\nG (glob threads) | L (loc size) | Atomic? | Time (s) | Result | Valid")
    print("----------------------------------------------------------------------------")
    
    zero_buf = np.array([0], dtype=np.int32)
    
    for G, L in test_cases:
        if G < L or G > N:
            continue
            
        for use_atomic in [1, 0]:
            cl.enqueue_copy(queue, d_result, zero_buf)
            queue.finish()
            
            try:
                start_time = time.time()
                if L == 1:
                    program.count_primes(
                        queue, (G,), None,
                        d_array, d_result, np.int32(N), np.int32(use_atomic))
                else:
                    program.count_primes(
                        queue, (G,), (L,),
                        d_array, d_result, np.int32(N), np.int32(use_atomic))
                queue.finish()
                elapsed = (time.time() - start_time)
                
                # Get result
                result = np.zeros(1, dtype=np.int32)
                cl.enqueue_copy(queue, result, d_result)
                queue.finish()
                
                valid = "YES" if result[0] == cpu_result else "NO"
                atomic_str = "YES" if use_atomic else "NO"
                
                print(f"{G:^18} | {L:^13} | {atomic_str:^7} | {elapsed:>8.6f} | {result[0]:>6} | {valid}")
                
            except Exception as e:
                atomic_str = "YES" if use_atomic else "NO"
                print(f"{G:^18} | {L:^13} | {atomic_str:^7} | {'ERROR':>8} | {'--':>6} | --")
                print(f"  Error: {str(e)}")

if __name__ == "__main__":
    main()