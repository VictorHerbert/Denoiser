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

float waveletfilterPixel(uint3 pos, Mat3D<float> in, Mat3D<float> out, Mat3D<float> albedo, Mat3D<float> normal,
    int kerSize, int offset, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal);

void waveletfilterCPU(Mat3D<float> in, Mat3D<float> out, Mat3D<float> albedo, Mat3D<float> normal,
    int kerSize, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal);

void waveletfilterGPU(Mat3D<float> in, Mat3D<float> out, Mat3D<float> albedo, Mat3D<float> normal,
    int kerSize, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal);

#endif
