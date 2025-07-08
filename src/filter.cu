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
    int3 pos = {0,0,0};
    do {
        oriSum[pos.z] += original[pos]*original[pos];
        distSum[pos.z] += (original[pos] - noisy[pos])*(original[pos] - noisy[pos]);

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

    Mat3D<float> buffer[2] = {in, out};
    int depth = 4;
    for(int i = 0; i < depth; i++){
        int3 pos = {0,0,0};
        do {
            out[pos] = waveletfilterPixel(pos, buffer[i%2], buffer[(i+1)%2], albedo, normal, kerSize, 1<<i, sigmaSpace, sigmaColor, sigmaAlbedo, sigmaNormal);
        } while(in.advanceIterator(pos));
    }

    in = buffer[depth%2];
    out = buffer[(depth+1)%2];
}

float waveletfilterPixel(int3 pos, Mat3D<float> in, Mat3D<float> out, Mat3D<float> albedo, Mat3D<float> normal,
    int kerSize, int offset, float sigmaSpace, float sigmaColor, float sigmaAlbedo, float sigmaNormal)
{
    kerSize = 5;//TODO generalize (or not)
    float acum = 0;
    float normFactor = 0;
    int halfSize = kerSize/2;
    float h[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0}; // Coefs of Pascal TriaNGLE
    int2 d, n;
    for(d.x = -halfSize; d.x <= halfSize; d.x++){
        for(d.y = -halfSize; d.y <= halfSize; d.y++){
            n.x = pos.x + d.x * offset;
            n.y = pos.y + d.y * offset;

            if( n.x >= 0 && n.x < in.size.x &&
                n.y >= 0 && n.y < in.size.y ){
                float3 dcol    = in[n] - in[pos];
                float3 dAlbedo = albedo[n] - albedo[pos];
                //float3 dNormal = 1.0 - normal[n]*normal[pos];
                float wWavelet = h[abs(d.x)] * h[abs(d.y)];

                float w =
                    wWavelet *
                    gaussian(make_float2(d), sigmaSpace) * // Simplify using exp(a) * exp(b) = exp(a + b)
                    gaussian(dcol, sigmaColor) *
                    gaussian(dAlbedo, sigmaAlbedo);

                acum += in[make_int3(n.x,n.y,pos.z)] * w;
                normFactor += w;
            }
        }
    }
    acum /= normFactor;
    return acum;

}