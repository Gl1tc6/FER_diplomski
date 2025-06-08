import numpy as np
import pyopencl as cl
import time
import math

def test_basic_opencl():
    """Test if basic OpenCL operations work at all"""
    print("=== Testing Basic OpenCL Functionality ===")
    
    # Initialize OpenCL
    platforms = cl.get_platforms()
    if not platforms:
        print("No OpenCL platforms found!")
        return False
        
    devices = []
    for platform in platforms:
        try:
            devices.extend(platform.get_devices())
        except:
            pass
            
    if not devices:
        print("No OpenCL devices found!")
        return False
        
    device = devices[0]
    print(f"Using device: {device.name}")
    
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    # Extremely simple kernel - just copy data
    simple_kernel = """
    __kernel void simple_copy(__global const int* input, __global int* output) {
        int gid = get_global_id(0);
        output[gid] = input[gid] + 1;
    }
    """
    
    try:
        program = cl.Program(ctx, simple_kernel).build()
        print("✓ Simple kernel compiled")
        
        # Test with small array
        test_data = np.array([1, 2, 3, 4], dtype=np.int32)
        
        mf = cl.mem_flags
        d_input = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_data)
        d_output = cl.Buffer(ctx, mf.WRITE_ONLY, test_data.nbytes)
        
        # Run with single thread
        program.simple_copy(queue, (1,), None, d_input, d_output)
        queue.finish()
        
        result = np.zeros_like(test_data)
        cl.enqueue_copy(queue, result, d_output)
        
        print(f"✓ Simple kernel executed: {test_data[0]} -> {result[0]}")
        return True
        
    except Exception as e:
        print(f"✗ Simple kernel failed: {e}")
        return False

def try_cpu_device():
    """Try using CPU OpenCL device if available"""
    print("\n=== Trying CPU OpenCL Device ===")
    
    platforms = cl.get_platforms()
    cpu_device = None
    
    for platform in platforms:
        try:
            cpu_devices = platform.get_devices(device_type=cl.device_type.CPU)
            if cpu_devices:
                cpu_device = cpu_devices[0]
                break
        except:
            continue
    
    if not cpu_device:
        print("No CPU OpenCL device found")
        return False
        
    print(f"Found CPU device: {cpu_device.name}")
    
    try:
        ctx = cl.Context([cpu_device])
        queue = cl.CommandQueue(ctx)
        
        # Simple prime counting kernel for CPU
        kernel_code = """
        bool is_prime(int n) {
            if (n < 2) return false;
            if (n == 2) return true;
            if (n % 2 == 0) return false;
            for (int i = 3; i * i <= n; i += 2) {
                if (n % i == 0) return false;
            }
            return true;
        }

        __kernel void count_primes(__global const int* array, __global int* result, const int n) {
            int gid = get_global_id(0);
            if (gid != 0) return;  // Only first thread works
            
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (is_prime(array[i])) count++;
            }
            result[0] = count;
        }
        """
        
        program = cl.Program(ctx, kernel_code).build()
        
        N = 100  # Small test
        host_array = np.arange(N, dtype=np.int32)
        
        mf = cl.mem_flags
        d_array = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_array)
        d_result = cl.Buffer(ctx, mf.WRITE_ONLY, 4)
        
        program.count_primes(queue, (1,), None, d_array, d_result, np.int32(N))
        queue.finish()
        
        result = np.zeros(1, dtype=np.int32)
        cl.enqueue_copy(queue, result, d_result)
        
        print(f"✓ CPU OpenCL worked: found {result[0]} primes in 0-{N-1}")
        return True
        
    except Exception as e:
        print(f"✗ CPU OpenCL failed: {e}")
        return False

def main():
    print("OpenCL GPU Crash Debugging")
    print("=" * 50)
    
    # Step 1: Test if basic OpenCL works
    if not test_basic_opencl():
        print("\n❌ Basic OpenCL is broken - likely driver issue")
        print("\nTroubleshooting suggestions:")
        print("1. Update AMD GPU drivers")
        print("2. Install mesa-opencl-icd: sudo apt install mesa-opencl-icd")
        print("3. Try: sudo apt install amdgpu-dkms")
        print("4. Check dmesg for GPU errors: dmesg | grep -i amdgpu")
        return
    
    # Step 2: Try CPU device as fallback
    if try_cpu_device():
        print("\n✓ CPU OpenCL works - GPU-specific issue")
    
    # Step 3: Try minimal GPU computation
    print("\n=== Minimal GPU Prime Test ===")
    
    platforms = cl.get_platforms()
    devices = []
    for platform in platforms:
        try:
            gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
            devices.extend(gpu_devices)
        except:
            pass
    
    if not devices:
        print("No GPU devices found")
        return
        
    device = devices[0]
    print(f"Testing GPU: {device.name}")
    
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    # Minimal prime kernel - no loops, just check a few numbers
    minimal_kernel = """
    __kernel void minimal_prime_check(__global const int* input, __global int* output) {
        int gid = get_global_id(0);
        int n = input[gid];
        
        // Hardcode checks for small numbers to avoid loops
        if (n == 2 || n == 3 || n == 5 || n == 7 || n == 11 || n == 13 || n == 17 || n == 19 || n == 23) {
            output[gid] = 1;
        } else if (n < 2 || n % 2 == 0 || n % 3 == 0 || n % 5 == 0) {
            output[gid] = 0;
        } else {
            output[gid] = 1;  // Assume prime for other small numbers
        }
    }
    """
    
    try:
        program = cl.Program(ctx, minimal_kernel).build()
        
        # Test with just a few numbers
        test_nums = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int32)
        
        mf = cl.mem_flags
        d_input = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_nums)
        d_output = cl.Buffer(ctx, mf.WRITE_ONLY, test_nums.nbytes)
        
        # Single thread
        program.minimal_prime_check(queue, (1,), None, d_input, d_output)
        queue.finish()
        
        result = np.zeros_like(test_nums)
        cl.enqueue_copy(queue, result, d_output)
        
        print(f"✓ Minimal GPU test worked!")
        print(f"Input:  {test_nums}")
        print(f"Output: {result}")
        
        # If this works, the issue is with loops or complex logic
        print("\n→ The crash is likely caused by loops in the prime checking function")
        print("→ Your AMD GPU driver may have issues with complex kernels")
        
    except Exception as e:
        print(f"✗ Even minimal GPU kernel failed: {e}")
        print("\n❌ Your AMD GPU OpenCL driver appears to be fundamentally broken")
        print("\nSuggestions:")
        print("1. Use CPU OpenCL instead")
        print("2. Update to latest AMD drivers")
        print("3. Check if GPU is overheating: sensors")
        print("4. Try different OpenCL implementation")

if __name__ == "__main__":
    main()