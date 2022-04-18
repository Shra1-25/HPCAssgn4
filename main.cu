#include <iostream>
#include "jacobi2D.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <string>

//! Harmonic function of x and y is used to compute true start_u and the boundary start_u. 
// __host__
// double getHarmonic(double x, double y) 
// {
//     //! Real Part ((z - 0.5 - 0.5i)^5) = (x - 0.5)^5 - 10 (x - 0.5)^3 (y - 0.5)^2 + 5 (x - 0.5) (y - 0.5)^4.

//     x -= 0.5;
//     y -= 0.5;
//     return pow(x, 5) - 10 * pow(x, 3) * pow(y, 2) + 5 * pow(x, 1) * pow(y, 4);
// }

/*! Compute order of N^2 Jacobi iterations for harmonic solution on xy unit square for boundary start_u where
 *      we divide the square into an NxN grid; save the results to file. 
 * 
 *  The number N is sent to the executable as a string and as the first and only parameter. The default value
 *      is 20 if no parameter is given. Also we require N > 1. 
 */
void swap(double **r, double **s)
{
    double *pSwap = *r;
    *r = *s;
    *s = pSwap;
}

int main(int argc, char * argv[]) 
{
    int N;
    if(argc < 2) N = 100; // default if no value given
    else N = std::stoi(argv[1]);

    int max_iter;
    if(argc < 3) max_iter = 1000; // default
    else max_iter = std::stoi(argv[2]);

    int dimensions[2] = {N, N}, // The dimensions of the grid
        nthreads = N / 10 + 1, // Number of CUDA threads per CUDA block dimension. 
        // nthreads = N,
        u_mem_required = dimensions[0] * dimensions[1] * sizeof(double);
    
    // const double bottom_left[2] = {0, 0}, top_right[2] = {1, 1}; // diagonal end points

    double * start_u, * f, * u, * u_new, * f_device;
    const dim3 blockSize( nthreads , nthreads), // The size of CUDA block of threads.
               gridSize( (dimensions[0] + nthreads - 1) / nthreads, (dimensions[1] + nthreads - 1) / nthreads);
    // const dim3 blockSize( 10 , 10), // The size of CUDA block of threads.
            //    gridSize( dimensions[0] / 10, dimensions[1] / 10 );

    std::cout << "Set u with 0s and f with 1s" << std::endl;
    
    // Initialize u and f
    start_u = new double[dimensions[0] * dimensions[1]];
    f = new double[dimensions[0] * dimensions[1]];
    for(int i = 0; i < dimensions[0]; i++) {
        int offset = i * dimensions[1];
        for(int j = 0; j < dimensions[1]; j++) {
            start_u[offset + j] = 0;
            f[offset + j] = 1;
        }
    }

    std::cout << "Initial residual = " << calculate_residual(start_u, f, dimensions) << std::endl;

    // Need to copy start_u from host to CUDA device.
    //std::cout << "Copying contents from host to Device" << std::endl;
    try 
    {
        copyToDevice(start_u, f, dimensions, &u, &u_new, &f_device);
    }
    catch( ... )
    {
        std::cout << "Caused Exception while copying to device" << std::endl;
    }

    //double h = 1.0 / (N + 1);
    //double h2 = h*h;
    //std::cout << "Initiating Jacobi iterations" << std::endl;
    for( int i = 0; i < max_iter; i++)
    {
        // Call CUDA device kernel to a Jacobi iteration. 
        doJacobiIteration<<< gridSize, blockSize >>>(dimensions[0], dimensions[1], u, u_new, f_device);
        cudaDeviceSynchronize();
        if(cudaGetLastError() != cudaSuccess)
        {
            std::cout << "Error Launching Kernel" << std::endl;
            return 1;
        }
        // swap(&u, &u_new);
        cudaMemcpyAsync( u, u_new, u_mem_required, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        // cudaMemcpy( start_u, u, u_mem_required, cudaMemcpyDeviceToHost );
        
    }

    // Get the result from the CUDA device.
    //std::cout << "Copying result back to start_u" << std::endl;
    if(cudaMemcpy( start_u, u, u_mem_required, cudaMemcpyDeviceToHost ) != cudaSuccess) 
    {
        std::cout << "There was a problem retrieving the result from the device" << std::endl;
        return 1;    
    }

    /*std::cout << "Final u\n";
    for(int j = 0; j < N; j++)
        std::cout << start_u[j] << ' ';
    std::cout << '\n';*/
    
    // Final residual
    std::cout << "Final residual = " << calculate_residual(start_u, f, dimensions) << std::endl;

    // Clean up memory.
    cudaFree(u);
    cudaFree(u_new);
    delete [] start_u;
    delete [] f;

    return 0;
}
