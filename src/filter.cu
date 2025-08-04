#include "filter.cuh"

#include <math.h>
#include "third_party/helper_math.h"
#include "iostream"


int idxToKernel(int d, int kernelSize){
    return d - kernelSize/2;
}

int2 idxToKernel(int2 d, int kernelSize){
    return {idxToKernel(d.x, kernelSize), idxToKernel(d.y, kernelSize)};
}

float gaussian(float p, float sigma){
    return exp(-p/(2*sigma*sigma));
}

float gaussian(float2 p, float sigma){
    return exp(-(p.x*p.x+p.y*p.y)/(2*sigma*sigma));
}

float gaussian(float3 p, float sigma){
    return exp(-(p.x*p.x+p.y*p.y+p.z*p.z)/(2*sigma*sigma));
}

float3 log10(float3 f){
    return {log10(f.x), log10(f.y), log10(f.z)};
}

float3 snrCPU(float3* original, float3* noisy, int2 size){
    float3 oriSum = {0,0,0};
    float3 distSum = {0,0,0};
    int2 pos = {0,0};
    do {
        oriSum += original[index(pos, size)]*original[index(pos, size)];
        distSum += (original[index(pos, size)] - noisy[index(pos, size)])*(original[index(pos, size)] - noisy[index(pos, size)]);
    } while(advanceIterator(pos, size));

    return 10.0 * log10(oriSum/distSum);
}


inline bool advanceIterator(int2& pos, int2 size) {
    pos.y++;
    if (pos.y < size.y)
        return true;

    pos.y = 0;
    pos.x++;
    if (pos.x < size.x)
        return true;

    return false;
}

void waveletfilterCPU(float3* in, float3* out, float3* albedo, float3* normal, int2 shape,
    int kerSize, int depth, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal)
{

    float3* buffer[2] = {in, out};
    for(int i = 0; i < depth; i++){
        int2 pos = {0,0};
        do {
            waveletfilterPixel(pos, buffer[i%2], buffer[(i+1)%2], nullptr, albedo, normal, shape, kerSize, 1<<i, sigmaSpace, sigmaColor, sigmaAlbedo, sigmaNormal);
        } while(advanceIterator(pos, shape));
    }

    in = buffer[depth%2];
    out = buffer[(depth+1)%2];
}

void waveletfilterGPU(float3* in, float3* out, float3* albedo, float3* normal, int2 shape,
    int kerSize, int depth, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal)
{

    //CudaVector<float> vVariance(totalSize(shape));
    CudaVector<float3> vAlbedo(albedo, totalSize(shape));
    CudaVector<float3> vNormal(normal, totalSize(shape));

    dim3 blockSize(16, 16);
    dim3 gridSize((shape.x + 15) / 16, (shape.y + 15) / 16);

    CudaVector<float3> b1(in, totalSize(shape));
    CudaVector<float3> b2(totalSize(shape));



    float3* buffer[2] = {b1.data, b2.data};
    for(int i = 0; i < depth; i++){
        waveletKernel<<<gridSize,blockSize>>>(buffer[i%2], buffer[(i+1)%2], nullptr, vAlbedo.data, vNormal.data, shape, kerSize, 1<<i, sigmaSpace, sigmaColor/(1<<i), sigmaAlbedo, sigmaNormal);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(out, buffer[depth%2], sizeof(float3)*totalSize(shape), cudaMemcpyDeviceToHost);
}

__global__ void waveletKernel(float3* in, float3* out, float* variance, float3* albedo, float3* normal, int2 shape,
    int kerSize, int offset, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal){
    
    int2 pos = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };

    if(index(pos, shape) >= totalSize(shape)){
        return;
    }
    out[index(pos, shape)] = waveletfilterPixel(pos, in, out, variance, albedo, normal, shape, kerSize, offset, sigmaSpace, sigmaColor, sigmaAlbedo, sigmaNormal);
}

float lum(float3 col){
    return 0.2126 *col.x + 0.7152*col.y + 0.0722*col.z;
}

float3 waveletfilterPixel(int2 pos, float3* in, float3* out, float* variance, float3* albedo, float3* normal, int2 shape,
    int kerSize, int offset, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal)
{
    float3 colAcum = {0, 0, 0};
    float normFactor = 1e-6;
    int halfSize = kerSize/2;
    float h[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0}; // Coefs of Pascal TriaNGLE
    int2 d, n;
    float var = 10;//variance[index(pos, shape)];


    float lp = lum(in[index(pos, shape)]);

    for(d.x = -halfSize; d.x <= halfSize; d.x++){
        for(d.y = -halfSize; d.y <= halfSize; d.y++){
            n.x = pos.x + d.x * offset;
            n.y = pos.y + d.y * offset;

            if( n.x >= 0 && n.x < shape.x &&
                n.y >= 0 && n.y < shape.y ){
                float ln = lum(in[index(n, shape)]);
                float dcol = abs(lp-ln);
                float2 dSpace = make_float2(d*offset);
                float3 dAlbedo = {0,0,0};//albedo[index(n, shape)] - albedo[index(pos, shape)];

                float wCol = gaussian(dcol, sqrt(var)*sigmaColor);
                float wSpace = gaussian(dSpace, sigmaSpace);
                float wAlbedo = gaussian(dAlbedo, sigmaAlbedo);
                float wWavelet = h[abs(d.x)] * h[abs(d.y)];

                float w =
                    wWavelet
                    * wSpace // Simplify using exp(a) * exp(b) = exp(a + b)
                    * wCol
                    * wAlbedo;

                colAcum += in[index(n, shape)] * w;
                normFactor += w;
            }
        }
    }
    colAcum /= normFactor;
    out[index(pos, shape)] = colAcum;

    return colAcum;
}