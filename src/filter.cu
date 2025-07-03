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

float3 snrCPU(Mat3D<float> original, Mat3D<float> noisy){
    if(original.size != noisy.size)
        throw std::runtime_error("Matrix sizes differ");

    float oriSum[3] = {0,0,0};
    float distSum[3] = {0,0,0};
    uint3 pos = {0,0,0};
    do {
        oriSum[pos.z] += original(pos)*original(pos);
        distSum[pos.z] += (original(pos) - noisy(pos))*(original(pos) - noisy(pos));
        
    } while(original.advanceIterator(pos));

    return {
        10 * log10(oriSum[0]/distSum[0]),
        10 * log10(oriSum[1]/distSum[1]),
        10 * log10(oriSum[2]/distSum[2])
    };
    
}

void crossBilateralfilterCPU(Mat3D<float> in, Mat3D<float> out, Mat3D<float> aux_buffer,
    int kerSize, float sigmaSpace, float sigmaColor, float sigmaAux)
{

    if(in.size.x != out.size.x || in.size.y != out.size.y || in.size.z != out.size.z)
        throw std::runtime_error("Matrix sizes differ");

    uint3 pos = {0,0,0};
    do {
        out(pos.x, pos.y, pos.z) = crossBilateralfilterPixel(pos, in, out, aux_buffer, kerSize, sigmaSpace, sigmaColor, sigmaAux);
    } while(in.advanceIterator(pos));

}

float crossBilateralfilterPixel(uint3 pos, Mat3D<float> in, Mat3D<float> out, Mat3D<float> aux_buffer,
    int kerSize, float sigmaSpace, float sigmaColor, float sigmaAux)
{
    float acum = 0;
    float normFactor = 0;
    for(int dx = -kerSize/2; dx <= kerSize/2; dx++){
        for(int dy = -kerSize/2; dy <= kerSize/2; dy++){
            int nx = pos.x + dx;
            int ny = pos.y + dy;
        
            if( nx >= 0 && nx < in.size.x &&
                ny >= 0 && ny < in.size.y ){
                float3 dcol = make_float3(in(nx,ny,0), in(nx,ny,1), in(nx,ny,2)) - make_float3(in(pos.x,pos.y,0), in(pos.x,pos.y,1), in(pos.x,pos.y,2));
                float w = gaussian(make_float2(dx,dy), 5) * gaussian(dcol, .1);
                acum += in(nx,ny,pos.z) * w;
                normFactor += w;
            }   
        }
    }
    acum /= normFactor;
    return acum;

}