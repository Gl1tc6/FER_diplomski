#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

void matrixMultiplyGPU(int n, int **A, int **B, int **C) {
    // Pretvaranje matrica iz 2D u 1D za OpenCL
    int *A_flat = (int *)malloc(n * n * sizeof(int));
    int *B_flat = (int *)malloc(n * n * sizeof(int));
    int *C_flat = (int *)malloc(n * n * sizeof(int));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_flat[i * n + j] = A[i][j];
            B_flat[i * n + j] = B[i][j];
            C_flat[i * n + j] = 0;
        }
    }

    // OpenCL inicijalizacija
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    cl_int err;
    err = clGetPlatformIDs(1, &platform, NULL);
    err |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    // Učitavanje OpenCL jezgrene funkcije
    const char *kernelSource =
        "__kernel void matrixMultiplyKernel(__global int *A, __global int *B, __global int *C, const int N) {"
        "    int row = get_global_id(0);"
        "    int col = get_global_id(1);"
        "    int sum = 0;"
        "    for (int i = 0; i < N; i++) {"
        "        sum += A[row * N + i] * B[i * N + col];"
        "    }"
        "    C[row * N + col] = sum;"
        "}";
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    err |= clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "matrixMultiplyKernel", &err);

    // Priprema OpenCL memorijskih objekata
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, n * n * sizeof(int), NULL, &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, n * n * sizeof(int), NULL, &err);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * n * sizeof(int), NULL, &err);

    // Kopiranje podataka u GPU memoriju
    err |= clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, n * n * sizeof(int), A_flat, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, n * n * sizeof(int), B_flat, 0, NULL, NULL);

    // Postavljanje argumenata jezgrene funkcije
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &n);

    // Postavljanje globalnih i lokalnih dimenzija
    size_t globalSize[2] = {n, n};
    err |= clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);

    // Dohvaćanje rezultata
    err |= clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, n * n * sizeof(int), C_flat, 0, NULL, NULL);

    // Pretvaranje natrag u 2D matricu
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = C_flat[i * n + j];
        }
    }

    // Oslobađanje resursa
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(A_flat);
    free(B_flat);
    free(C_flat);
}

int main() {
    int n = 3;
    int **A = (int **)malloc(n * sizeof(int *));
    int **B = (int **)malloc(n * sizeof(int *));
    int **C = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        A[i] = (int *)malloc(n * sizeof(int));
        B[i] = (int *)malloc(n * sizeof(int));
        C[i] = (int *)malloc(n * sizeof(int));
    }

    // Inicijalizacija matrica A i B
    int counter = 1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = counter;
            B[i][j] = counter + 1;
            counter++;
        }
    }

    // Poziv funkcije za množenje matrica na GPU-u
    matrixMultiplyGPU(n, A, B, C);

    // Ispis rezultata
    printf("Matrix A:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", A[i][j]);
        }
        printf("\n");
    }

    printf("Matrix B:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", B[i][j]);
        }
        printf("\n");
    }

    printf("Matrix C (Result):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    // Oslobađanje memorije
    for (int i = 0; i < n; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}
