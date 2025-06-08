import numpy as np
import time
import math
import sys
import argparse
import pyopencl as cl
import warnings
warnings.filterwarnings("ignore", message="Non-empty compiler output")

kernel_source = """
__kernel void compute_pi_partial(__global double* results, 
                                const int start_index, 
                                const int elements_per_thread) {
    int gid = get_global_id(0);
    int local_start = start_index + gid * elements_per_thread;
    
    double sum = 0.0;
    for(int i = 0; i < elements_per_thread; i++) {
        int n = local_start + i;
        double term = (n % 2 == 0) ? 1.0 : -1.0;
        sum += term / (2.0 * n + 1.0);
    }
    
    results[gid] = sum;
}
"""

def get_opencl_devices():
    devices = []
    try:
        platforms = cl.get_platforms()
        for platform in platforms:
            for device in platform.get_devices():
                devices.append({
                    'platform': platform.name,
                    'device': device,
                    'name': device.name,
                    'type': cl.device_type.to_string(device.type)
                })
    except Exception as e:
        print(e)
    
    return devices

def compute_pi_sequential(N: int) -> float:
    sum_val = 0.0
    for i in range(N):
        term = (-1)**i / (2*i + 1)
        sum_val += term
    return 4.0 * sum_val

def compute_pi_opencl(N: int, M: int, L: int, device=None) -> tuple:
    if device is None:
        devices = get_opencl_devices()
        device_name = devices[0]['name']
    else:
        device_name = device.name
    
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    
    # Kompajliranje kernela
    program = cl.Program(context, kernel_source).build()
    kernel = program.compute_pi_partial
    
    num_work_items = (N + M - 1) // M
    num_work_groups = (num_work_items + L - 1) // L
    global_size = num_work_groups * L
    local_size = L
    
    print(f"  Uređaj: {device_name}")
    print(f"  Broj radnih jedinica: {num_work_items}")
    print(f"  Broj radnih grupa: {num_work_groups}")
    print(f"  Globalna veličina: {global_size}")
    print(f"  Lokalna veličina: {local_size}")
    
    # mem alloc
    results_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, 
                              size=global_size * np.dtype(np.float64).itemsize)
    
    start_time = time.time()
    
    # Pokretanje kernela za različite dijelove posla
    total_sum = 0.0
    processed = 0
    
    while processed < N:
        remaining = N - processed
        current_work_items = min(num_work_items, (remaining + M - 1) // M)
        current_global_size = ((current_work_items + L - 1) // L) * L
        
        kernel(queue, (current_global_size,), (local_size,), 
               results_buffer, np.int32(processed), np.int32(M))
        
        queue.finish()
        
        results = np.empty(current_global_size, dtype=np.float64)
        cl.enqueue_copy(queue, results, results_buffer)
        
        # valid
        for i in range(current_work_items):
            total_sum += results[i]
        
        processed += current_work_items * M
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    pi_value = 4.0 * total_sum
    
    device_info = {
        'name': device_name,
        'type': cl.device_type.to_string(device.type),
        'global_size': global_size,
        'local_size': local_size,
        'work_groups': num_work_groups
    }
    
    return pi_value, execution_time, device_info

def main():
    parser = argparse.ArgumentParser(description='Računanje PI-a')
    parser.add_argument('N', type=int, help='Ukupan broj elemenata reda')
    parser.add_argument('M', type=int, help='Broj elemenata po radnoj jedinici')
    parser.add_argument('L', type=int, help='Veličina radne grupe')
    parser.add_argument('-d', '--device', type=int, default=0,
                       help='Indeks OpenCL uređaja (zadano: 0)')
    
    args = parser.parse_args()
    
    N, M, L = args.N, args.M, args.L
    
    print(f"Računanje Pi s N={N}, M={M}, L={L}")
    print("-" * 50)
    
    # Dohvaćanje uređaja
    devices = get_opencl_devices()
    if not devices:
        print("Nema dostupnih OpenCL uređaja")
        return
    
    if args.device >= len(devices):
        print(f"Uređaj {args.device} ne postoji. Dostupni uređaji: 0-{len(devices)-1}")
        return
    
    selected_device = devices[args.device]['device']
    
    print("Sekvencijalno računanje...")
    start_seq = time.time()
    pi_sequential = compute_pi_sequential(N)
    end_seq = time.time()
    time_sequential = end_seq - start_seq
    
    print(f"Sekvencijalni rezultat: Pi = {pi_sequential:.10f} ({time_sequential:.4f} s)")
    

    try:
        pi_opencl, time_opencl, device_info = compute_pi_opencl(N, M, L, selected_device)
        
        # Analiza rezultata
        speedup = time_sequential / time_opencl if time_opencl > 0 else 0
        error_percentage = abs(pi_opencl - math.pi) / math.pi * 100
        
        print(f"\nRezultati:")
        print(f"OpenCL rezultat:     Pi = {pi_opencl:.10f} ({time_opencl:.4f} s)")
        print(f"Točna vrijednost:    Pi = {math.pi:.10f}")
        print(f"Ubrzanje:            {speedup:.2f}x")
        print(f"Greška:              {error_percentage:.6f}%")
        
            
    except Exception as e:
        print(e)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("python3 2zad_OpenCL.py <N> <M> <L> [-d DEVICE]")
        sys.exit(1)
    
    main()