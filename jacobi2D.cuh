/*! jacobi.cuh
 * 
 * Functions for handling Jacobi iterations on CUDA device.
 *
 */

#pragma once

__global__ 
void doJacobiIteration(int dimX, int dimY, double* u, double* u_new, double* f);

__host__
void copyToDevice(double * start_u, double * f, const int dimensions[2], double** u, double** u_new, double ** f_device);

__host__
double calculate_residual(const double* u, const double* f, const int dimensions[2]);

__host__ 
void printValues(const int dimensions[2], const double * values);
