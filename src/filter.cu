#include "filter.cuh"

#include <math.h>
#include "third_party/helper_math.h"


int idxToKernel(int d, int kernelSize){
    return d - kernelSize/2;
}

int2 idxToKernel(int2 d, int kernelSize){
    return {idxToKernel(d.x, kernelSize), idxToKernel(d.y, kernelSize)};
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
            out[index(pos, shape)] =
                waveletfilterPixel(pos, buffer[i%2], buffer[(i+1)%2], albedo, normal, shape, kerSize, 1<<i, sigmaSpace, sigmaColor, sigmaAlbedo, sigmaNormal);
        } while(advanceIterator(pos, shape));
    }

    in = buffer[depth%2];
    out = buffer[(depth+1)%2];
}

void waveletfilterGPU(float3* in, float3* out, float3* albedo, float3* normal, int2 size,
    int kerSize, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal)
{
    float* buffer[2];
    
    //cudaMalloc(&buffer[0], );

}

__global__ void waveletKernel(float3* in, float3* out, float3* albedo, float3* normal, int2 size,
    int kerSize, int offset, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal){

}

float3 waveletfilterPixel(int2 pos, float3* in, float3* out, float3* albedo, float3* normal, int2 size,
    int kerSize, int offset, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal)
{    
    float3 acum = {0, 0, 0};
    float normFactor = 0;
    int halfSize = kerSize/2;
    float h[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0}; // Coefs of Pascal TriaNGLE
    int2 d, n;

    //float lp = 0.2126 * in[pos].x + 0.7152 * in[p].y + 0.0722 * in[p].z;

    for(d.x = -halfSize; d.x <= halfSize; d.x++){
        for(d.y = -halfSize; d.y <= halfSize; d.y++){
            n.x = pos.x + d.x * offset;
            n.y = pos.y + d.y * offset;

            if( n.x >= 0 && n.x < size.x &&
                n.y >= 0 && n.y < size.y ){
                float3 dcol    = in[index(n, size)] - in[index(pos, size)];
                float3 dAlbedo = albedo[index(n, size)] - albedo[index(pos, size)];
                //float3 dNormal = 1.0 - normal[n]*normal[pos];
                float wWavelet = h[abs(d.x)] * h[abs(d.y)];
                

                float w =
                    wWavelet 
                    //* gaussian(make_float2(d*offset), sigmaSpace) * // Simplify using exp(a) * exp(b) = exp(a + b)
                    * gaussian(dcol, sigmaColor);
                    //* gaussian(dAlbedo, sigmaAlbedo);

                acum += in[index(n, size)] * w;
                normFactor += w;
            }
        }
    }
    acum /= normFactor;
    return acum;

}