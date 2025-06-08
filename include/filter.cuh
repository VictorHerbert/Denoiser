#ifndef FILTER_H
#define FILTER_H

#include "matrix.cuh"

//float gaussian(float p, float sigma);
float gaussian(float2 p, float sigma);
float gaussian(float3 p, float sigma);

void gaussianFilterCPU(Mat3D<float> in, Mat3D<float> out, int kerSize = 5);
void gaussianFilterGPU(Mat3D<float> in, Mat3D<float> out);

#endif