/*! jacobi.cu
 */

#include "jacobi2D.cuh"
#include <iostream>
#include <fstream>

__global__
void doJacobiIteration(int dimX, int dimY, double* u, double* u_new, double* f) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x,
              j = blockIdx.y * blockDim.y + threadIdx.y;
    const int ind = i * dimY + j;
    double u01 = 0.0, u0_1 = 0.0, u10 = 0.0, u_10 = 0.0;
    double h = 1.0 / (dimX + 1);
    double h2 = h*h;

    if(i > 0)       u_10 = u[ind - dimY];
    if(i < dimX-1)  u10 = u[ind + dimY];
    if(j > 0)       u0_1 = u[ind - 1];
    if(j < dimY-1)  u01 = u[ind + 1];

    // u_new[ind] = 0.25 * (h2 * f[ind] + u[ind+1] + u[ind-1] + u[ind+dimY] + u[ind-dimY]);
    u_new[ind] = (h2 * f[ind] + u01 + u0_1 + u10 + u_10) / 4.0;
}

__host__
void copyToDevice(double* start_u, double* f, const int dimensions[2], double ** u, double ** u_new, double ** f_device) {
    const int u_mem_required = dimensions[0] * dimensions[1] * sizeof(double); 

    if (cudaMalloc( (void**) u, u_mem_required ) != cudaSuccess)
        throw "Can't allocate u on device.";

    if (cudaMalloc( (void**) u_new, u_mem_required ) != cudaSuccess)
        throw "Can't allocate u_new on device.";

    if (cudaMalloc( (void**) f_device, u_mem_required ) != cudaSuccess)
        throw "Can't allocate f on device.";

    if(cudaMemcpy( *u, start_u, u_mem_required, cudaMemcpyHostToDevice ) != cudaSuccess)
        throw "Can't copy start_u to in on device.";

    if(cudaMemcpy( *u_new, start_u, u_mem_required, cudaMemcpyHostToDevice ) != cudaSuccess)
        throw "Can't copy start_u to out on device.";

    if(cudaMemcpy( *f_device, f, u_mem_required, cudaMemcpyHostToDevice ) != cudaSuccess)
        throw "Can't copy f to in on device.";
 
}

__host__
double calculate_residual(const double * u, const double * f, const int dimensions[2]) {

    // Now get the average error.
    double res = 0, diff = 0;
    int N = dimensions[0];
    double h = 1.0 / (N + 1), h2 = h * h;
    int i,j, index;
    // first row:
    diff = f[0] + (-4 * u[0] + u[1] + u[N]) / h2;
    res += diff * diff;
    for(j = 1; j < N-1; j++) {
        diff = f[j] + (-4 * u[j] + u[j-1] + u[j+1] + u[j+N]) / h2;
        res += diff * diff;
    }
    diff = f[N-1] + (-4 * u[N-1] + u[N-2] + u[2*N-1]) / h2;
    res += diff * diff;

    // inner rows:
    for(i = 1; i < N - 1; i++){
        index = N * i;
        diff = f[index] + (u[index + 1] + u[index + N] + u[index - N] \
                    - 4 * u[index]) / h2;
        res += diff * diff;
        for(j = 1; j < N - 1; j++) {
            index = N * i + j;
            diff = f[index] + (u[index + 1] + u[index - 1] + u[index + N] \
                        + u[index - N] - 4 * u[index]) / h2;
            res += diff * diff;
        }
        index = N * i + (N - 1);
        diff = f[index] + (u[index - 1] + u[index - N] + u[index + N] \
                    - 4 * u[index]) / h2;
        res += diff * diff;
    }

    // last row:
    index = N*(N-1);
    diff = f[index] + (u[index + 1] + u[index - N] - 4 * u[index]) / h2;
    res += diff * diff;
    for(j = 1; j < N-1; j++) {
        index = N*(N-1) + j;
        diff = f[index] + (u[index - 1] + u[index + 1] + u[index - N] \
                    - 4 * u[index]) / h2;
        res += diff * diff;
    }
    index = N * N - 1;
    diff = f[index] + (u[index - 1] + u[index - N] - 4 * u[index]) / h2;
    res += diff * diff;
    
    return sqrt(res)/3.3;
}

__host__
void printValues(const int dimensions[2], const double * values)  {
    const double * pos = values;
    for (int i = 0; i < dimensions[0]; i++) 
    {
        for (int j = 0; j < dimensions[1]; j++, pos++)
            std::cout << *pos << ",\t";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
