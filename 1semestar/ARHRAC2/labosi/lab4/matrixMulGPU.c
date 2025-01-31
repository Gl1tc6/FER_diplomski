#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel za množenje matrica
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

// Pomoćna funkcija za provjeru OpenCL grešaka
void check_error(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL Error %d: %s\n", err, msg);
        exit(1);
    }
}

// Funkcija za množenje matrica na GPU
void gpu_matrix_multiply(float* A, float* B, float* C, int N) {
    cl_int err;
    
    // Dohvat platforme i uređaja
    cl_platform_id platform;
    cl_device_id device;
    err = clGetPlatformIDs(1, &platform, NULL);
    check_error(err, "Getting platform");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    check_error(err, "Getting device");
    
    // Stvaranje konteksta i reda naredbi
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_error(err, "Creating context");
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    check_error(err, "Creating command queue");
    
    // Stvaranje programa i prevođenje kernela
    cl_program program = clCreateProgramWithSource(context, 1, 
        (const char**)&matrixMultiplyKernel, NULL, &err);
    check_error(err, "Creating program");
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
            sizeof(buffer), buffer, &len);
        fprintf(stderr, "Build error: %s\n", buffer);
        exit(1);
    }
    
    // Stvaranje kernela
    cl_kernel kernel = clCreateKernel(program, "matrix_multiply", &err);
    check_error(err, "Creating kernel");
    
    // Alokacija memorijskih objekata
    size_t matrix_size = N * N * sizeof(float);
    cl_mem a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        matrix_size, A, &err);
    cl_mem b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        matrix_size, B, &err);
    cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        matrix_size, NULL, &err);
    
    // Postavljanje argumenata kernela
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &N);
    check_error(err, "Setting kernel arguments");
    
    // Pokretanje kernela
    size_t global_work_size[2] = { N, N };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size,
        NULL, 0, NULL, NULL);
    check_error(err, "Enqueueing kernel");
    
    // Čitanje rezultata
    err = clEnqueueReadBuffer(queue, c_mem, CL_TRUE, 0, matrix_size,
        C, 0, NULL, NULL);
    check_error(err, "Reading results");
    
    // Čišćenje
    clReleaseMemObject(a_mem);
    clReleaseMemObject(b_mem);
    clReleaseMemObject(c_mem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

// Primjer korištenja
int main() {
    const int N = 4; // Dimenzija matrica
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));
    float *C = (float*)malloc(N * N * sizeof(float));
    
    // Inicijalizacija matrica za testiranje
    for(int i = 0; i < N * N; i++) {
        A[i] = (float)(i + 1);
        B[i] = (float)(i + 1);
    }
    
    // Poziv GPU implementacije
    gpu_matrix_multiply(A, B, C, N);
    
    // Ispis rezultata
    printf("Rezultat množenja matrica:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%8.1f ", C[i * N + j]);
        }
        printf("\n");
    }
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}