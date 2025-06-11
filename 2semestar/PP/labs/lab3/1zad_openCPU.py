import numpy as np
import pyopencl as cl
import time
import math

def main():
    N = 2**21  # 4194304 elements
    print(f"# elements: {N}")

    platforms = cl.get_platforms()
    cpu_device = None
    for platform in platforms:
        for device in platform.get_devices():
            if device.type == cl.device_type.CPU:
                cpu_device = device
                break
        if cpu_device:
            break
    
    if not cpu_device:
        print("No CPU OpenCL device found!")
        return
    
    print(f"Using CPU: {cpu_device.name}")
    
    ctx = cl.Context([cpu_device])
    queue = cl.CommandQueue(ctx)
    
    kernel_code = """
    bool is_prime(int n) {
        if (n < 2) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        
        for (int i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) return false;
        }
        return true;
    }

    __kernel void count_primes(__global int* array, __global int* result, int n) {
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
        
        atomic_add(result, local_count);
    }
    """
    
    program = cl.Program(ctx, kernel_code).build()
    host_array = np.arange(N, dtype=np.int32)
    
    # Buffers
    d_array = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_array)
    d_result = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=4)
    
    # CPU reference
    print("CPU reference...")
    start_time = time.time()
    cpu_count = sum(1 for i in range(2, N) if all(i % j != 0 for j in range(2, int(math.sqrt(i)) + 1)) if i > 1)
    cpu_time = time.time() - start_time
    print(f"CPU: {cpu_count} primes ({cpu_time:.4f}s)")
    
    # Test different thread counts
    thread_counts = [1, 2, 4, 8, 16, 32]
    
    print("\nThreads | Time (s) | Result | Speedup")
    print("--------|----------|--------|--------")
    
    for threads in thread_counts:
        if threads > cpu_device.max_compute_units * 4:  # Reasonable limit
            continue
            
        zero = np.array([0], dtype=np.int32)
        cl.enqueue_copy(queue, d_result, zero)
        
        start_time = time.time()
        program.count_primes(queue, (threads,), None, d_array, d_result, np.int32(N))
        queue.finish()
        elapsed = time.time() - start_time
        
        result = np.zeros(1, dtype=np.int32)
        cl.enqueue_copy(queue, result, d_result)
        
        speedup = cpu_time / elapsed if elapsed > 0 else 0
        status = "OK" if result[0] == cpu_count else "ERR"
        res = result[0] if status == "ERR" else ""
        print(f"{threads:^7} | {elapsed:^8.4f} | {res} {status:^5} | {speedup:^6.2f}x")

if __name__ == "__main__":
    main()