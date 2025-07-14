#ifndef FILTER_H
#define FILTER_H

#include "matrix.cuh"

const int GAUSSIAN  = 0;
const int CROSS     = 1<<0;
const int BILATERAL = 1<<1;
const int WAVELET   = 1<<2;

float gaussian(float2 p, float sigma);
float gaussian(float3 p, float sigma);

float3 snrCPU(float3* original, float3* noisy, int2 shape);
float3 snrGPU(float3* original, float3* noisy, int2 shape);

void waveletfilterCPU(float3* in, float3* out, float3* albedo, float3* normal, int2 shape,
    int kerSize, int depth, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal);

void waveletfilterGPU(float3* in, float3* out, float3* albedo, float3* normal, int2 shape,
    int kerSize, int depth, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal);

__global__ void waveletKernel(float3* in, float3* out, float3* albedo, float3* normal, int2 shape,
    int kerSize, int offset, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal);

__host__ __device__ float3 waveletfilterPixel(int2 pos, float3* in, float3* out, float3* albedo, float3* normal, int2 shape,
    int kerSize, int offset, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal);

#endif
