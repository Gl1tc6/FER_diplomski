#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// Funkcija za mjerenje vremena u milisekundama
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0);
}

// CPU implementacija množenja matrica
void cpu_matrix_multiply(float* A, float* B, float* C, int N) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            float sum = 0.0f;
            for(int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// OpenCL kernel za množenje matrica
const char* matrixMultiplyKernel = "\n" \
"__kernel void matrix_multiply(__global float* A,                                \n" \
"                            __global float* B,                                 \n" \
"                            __global float* C,                                 \n" \
"                            const int N) {                                     \n" \
"    int row = get_global_id(0);                                               \n" \
"    int col = get_global_id(1);                                               \n" \
"                                                                              \n" \
"    float sum = 0.0f;                                                         \n" \
"    for (int k = 0; k < N; k++) {                                            \n" \
"        sum += A[row * N + k] * B[k * N + col];                              \n" \
"    }                                                                         \n" \
"    C[row * N + col] = sum;                                                  \n" \
"}                                                                             \n";

// Struktura za pohranu vremena profiliranja GPU implementacije
typedef struct {
    double transfer_to_gpu;
    double computation;
    double transfer_from_gpu;
    double total;
} GPUTiming;

// GPU implementacija s mjerenjem vremena
GPUTiming gpu_matrix_multiply(float* A, float* B, float* C, int N, cl_device_id device, 
                            cl_context context, cl_program program) {
    GPUTiming timing = {0, 0, 0, 0};
    cl_int err;
    double start_time;
    
    // Stvaranje reda naredbi
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    cl_kernel kernel = clCreateKernel(program, "matrix_multiply", &err);
    
    size_t matrix_size = N * N * sizeof(float);
    
    // Mjerenje vremena prijenosa na GPU
    start_time = get_time_ms();
    cl_mem a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        matrix_size, A, &err);
    cl_mem b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        matrix_size, B, &err);
    cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        matrix_size, NULL, &err);
    clFinish(queue);
    timing.transfer_to_gpu = get_time_ms() - start_time;
    
    // Postavljanje argumenata i pokretanje kernela
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &N);
    
    size_t global_work_size[2] = { N, N };
    
    // Mjerenje vremena računanja
    start_time = get_time_ms();
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size,
        NULL, 0, NULL, NULL);
    clFinish(queue);
    timing.computation = get_time_ms() - start_time;
    
    // Mjerenje vremena prijenosa s GPU-a
    start_time = get_time_ms();
    err = clEnqueueReadBuffer(queue, c_mem, CL_TRUE, 0, matrix_size,
        C, 0, NULL, NULL);
    clFinish(queue);
    timing.transfer_from_gpu = get_time_ms() - start_time;
    
    timing.total = timing.transfer_to_gpu + timing.computation + timing.transfer_from_gpu;
    
    // Čišćenje
    clReleaseMemObject(a_mem);
    clReleaseMemObject(b_mem);
    clReleaseMemObject(c_mem);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    
    return timing;
}

int main() {
    // Inicijalizacija OpenCL
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;
    
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_program program = clCreateProgramWithSource(context, 1, 
        (const char**)&matrixMultiplyKernel, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    // Veličine matrica za testiranje
    int sizes[] = {128, 256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("Dimenzija,CPU vrijeme (ms),GPU ukupno (ms),GPU→RAM (ms),Računanje (ms),RAM→GPU (ms)\n");
    
    for(int i = 0; i < num_sizes; i++) {
        int N = sizes[i];
        size_t matrix_size = N * N * sizeof(float);
        
        // Alokacija i inicijalizacija matrica
        float *A = (float*)malloc(matrix_size);
        float *B = (float*)malloc(matrix_size);
        float *C_cpu = (float*)malloc(matrix_size);
        float *C_gpu = (float*)malloc(matrix_size);
        
        // Slučajna inicijalizacija
        srand(time(NULL));
        for(int j = 0; j < N * N; j++) {
            A[j] = (float)rand() / RAND_MAX;
            B[j] = (float)rand() / RAND_MAX;
        }
        
        // CPU mjerenje
        double cpu_start = get_time_ms();
        cpu_matrix_multiply(A, B, C_cpu, N);
        double cpu_time = get_time_ms() - cpu_start;
        
        // GPU mjerenje
        GPUTiming gpu_timing = gpu_matrix_multiply(A, B, C_gpu, N, device, context, program);
        
        // Ispis rezultata
        printf("%d,%.2f,%.2f,%.2f,%.2f,%.2f\n", 
            N, cpu_time, gpu_timing.total, 
            gpu_timing.transfer_from_gpu, gpu_timing.computation, 
            gpu_timing.transfer_to_gpu);
        
        free(A);
        free(B);
        free(C_cpu);
        free(C_gpu);
    }
    
    // Čišćenje OpenCL resursa
    clReleaseProgram(program);
    clReleaseContext(context);
    
    return 0;
}