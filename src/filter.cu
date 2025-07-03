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


void waveletfilterCPU(Mat3D<float> in, Mat3D<float> out, Mat3D<float> albedo, Mat3D<float> normal,
    int kerSize, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal)
{
    if(in.size != out.size)
        throw std::runtime_error("Matrix sizes differ");

    uint3 pos = {0,0,0};
    do {        
        out(pos.x, pos.y, pos.z) = waveletfilterPixel(pos, in, out, albedo, normal, kerSize, 1<<1, sigmaSpace, sigmaColor, sigmaAlbedo, sigmaNormal);        
    } while(in.advanceIterator(pos));

}

float waveletfilterPixel(uint3 pos, Mat3D<float> in, Mat3D<float> out, Mat3D<float> albedo, Mat3D<float> normal,
    int kerSize, int offset, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal)
{
    float acum = 0;
    float normFactor = 0;
    int halfSize = kerSize/2;
    for(int dx = -halfSize; dx <= halfSize; dx++){
        for(int dy = -halfSize; dy <= halfSize; dy++){
            int nx = pos.x + dx * offset;
            int ny = pos.y + dy * offset;
        
            if( nx >= 0 && nx < in.size.x &&
                ny >= 0 && ny < in.size.y ){
                float3 dcol     = make_float3(in(nx,ny,0), in(nx,ny,1), in(nx,ny,2)) - make_float3(in(pos.x,pos.y,0), in(pos.x,pos.y,1), in(pos.x,pos.y,2));
                float3 dAlbedo  = make_float3(albedo(nx,ny,0), albedo(nx,ny,1), albedo(nx,ny,2)) - make_float3(albedo(pos.x,pos.y,0), albedo(pos.x,pos.y,1), albedo(pos.x,pos.y,2));                
                float3 dNormal  = 1-make_float3(albedo(nx,ny,0), albedo(nx,ny,1), albedo(nx,ny,2)) * make_float3(albedo(pos.x,pos.y,0), albedo(pos.x,pos.y,1), albedo(pos.x,pos.y,2));                

                float w =
                    gaussian(make_float2(dx,dy), sigmaSpace) * 
                    gaussian(dcol, sigmaColor) * 
                    gaussian(dAlbedo, sigmaAlbedo);
                acum += in(nx,ny,pos.z) * w;
                normFactor += w;
            }   
        }
    }
    acum /= normFactor;
    return acum;

}