#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#include "jacobi.h"

static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_mem psi_buf = NULL;
static cl_mem psitmp_buf = NULL;
static int opencl_ready = 0;

const char* kernel_source = 
"__kernel void jacobi_step(__global double* psinew, __global double* psi, int m, int n) {"
"    int i = get_global_id(0) + 1;"
"    int j = get_global_id(1) + 1;"
"    if (i <= m && j <= n) {"
"        int idx = i * (m + 2) + j;"
"        psinew[idx] = 0.25 * (psi[(i-1)*(m+2)+j] + psi[(i+1)*(m+2)+j] + "
"                              psi[i*(m+2)+(j-1)] + psi[i*(m+2)+(j+1)]);"
"    }"
"}";

void init_opencl(int m, int n) {
    cl_platform_id platform;
    cl_device_id device;
    cl_program program;
    cl_int err;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "jacobi_step", &err);
    
    size_t size = (m+2) * (n+2) * sizeof(double);
    psi_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
    psitmp_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
    
    clSetKernelArg(kernel, 2, sizeof(int), &m);
    clSetKernelArg(kernel, 3, sizeof(int), &n);
    
    clReleaseProgram(program);
    opencl_ready = (err == CL_SUCCESS) ? 1 : 0;
    
    if (opencl_ready) 
        printf("OpenCL accelerated Jacobi for %dx%d grid\n", m, n);
}

void jacobistep(double *psinew, double *psi, int m, int n) {
    if (!opencl_ready && context == NULL) {
        init_opencl(m, n);
    }
    
    // GPU
    if (opencl_ready) {
        size_t global[2] = {m, n};
        size_t local[2] = {16, 16};
        if (m < 16) local[0] = m;
        if (n < 16) local[1] = n;
        
        size_t size = (m+2) * (n+2) * sizeof(double);
        
        clEnqueueWriteBuffer(queue, psi_buf, CL_FALSE, 0, size, psi, 0, NULL, NULL);
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &psitmp_buf);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &psi_buf);
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, psitmp_buf, CL_TRUE, 0, size, psinew, 0, NULL, NULL);
        return;
    }
    
    // CPU fallback
    for(int i=1; i<=m; i++) {
        for(int j=1; j<=n; j++) {
            psinew[i*(m+2)+j] = 0.25*(psi[(i-1)*(m+2)+j] + psi[(i+1)*(m+2)+j] + 
                                     psi[i*(m+2)+j-1] + psi[i*(m+2)+j+1]);
        }
    }
}

double deltasq(double *newarr, double *oldarr, int m, int n) {
    double dsq = 0.0;
    for(int i=1; i<=m; i++) {
        for(int j=1; j<=n; j++) {
            double tmp = newarr[i*(m+2)+j] - oldarr[i*(m+2)+j];
            dsq += tmp*tmp;
        }
    }
    return dsq;
}

void jacobi_cleanup() {
    if (psi_buf) clReleaseMemObject(psi_buf);
    if (psitmp_buf) clReleaseMemObject(psitmp_buf);
    if (kernel) clReleaseKernel(kernel);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}