#include "filter.cuh"

#include <math.h>

int idxToKernel(int d, int kernelSize){
    return d - kernelSize/2;
}

int2 idxToKernel(int2 d, int kernelSize){
    return {idxToKernel(d.x, kernelSize), idxToKernel(d.y, kernelSize)};
}

float gaussian(float2 p, float sigma){
    return exp(-(p.x*p.x+p.y*p.y)/(2*sigma*sigma));
}

void gaussianFilterCPU(Mat3D<float> in, Mat3D<float> out, int kerSize){
    if(in.size.x != out.size.x || in.size.y != out.size.y || in.size.z != out.size.z)
        throw std::runtime_error("Matrix sizes differ");

    for(int ch = 0; ch < in.size.z; ch++){
        for(int x = 0; x < in.size.x; x++){
            for(int y = 0; y < in.size.y; y++){
            
                float acum = 0;
                float normFactor = 0;
                for(int dx = -kerSize/2; dx <= kerSize/2; dx++){
                    for(int dy = -kerSize/2; dy <= kerSize/2; dy++){
                        int nx = x + dx;
                        int ny = y + dy;
                    
                        if( nx >= 0 && nx < in.size.x &&
                            ny >= 0 && ny < in.size.y ){
                            acum += in(x,y,ch) * gaussian(make_float2(dx,dy), 1);
                            normFactor += gaussian(make_float2(dx,dy), 1);
                        }   
                    }
                }
                acum /= normFactor;
                out(x,y,ch) = acum;
            }
        }
    }
    
}