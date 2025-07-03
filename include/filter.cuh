#ifndef FILTER_H
#define FILTER_H

#include "matrix.cuh"

const int GAUSSIAN  = 0;
const int CROSS     = 1<<0;
const int BILATERAL = 1<<1;
const int WAVELET   = 1<<2;

float gaussian(float2 p, float sigma);
float gaussian(float3 p, float sigma);

float3 snrCPU(Mat3D<float> original, Mat3D<float> noisy);

float crossBilateralfilterPixel(uint3 pos, Mat3D<float> in, Mat3D<float> out, Mat3D<float> aux_buffer,
    int kerSize, float sigmaSpace, float sigmaColor, float sigmaAux);

void crossBilateralfilterCPU(Mat3D<float> in, Mat3D<float> out, Mat3D<float> aux_buffer,
    int kerSize, float sigmaSpace, float sigmaColor, float sigmaAux);

void crossBilateralfilterGPU(Mat3D<float> in, Mat3D<float> out, Mat3D<float> aux_buffer,
    int kerSize, float sigmaSpace, float sigmaColor, float sigmaAux);

#endif
