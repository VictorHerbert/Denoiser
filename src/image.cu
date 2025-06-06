#include "image.cuh"

#include <string>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

#include "third_party/helper_math.h"

#include <stdexcept>

Image::Image(int3 size){
    this->mat.size = size;
    vData.reserve(size.x * size.y * size.z);
    mat.data = vData.data();
    allocated = false;
}

Image::Image(std::string filename, bool grayscale){
    mat.data = (uchar*) stbi_load(filename.c_str(), &(mat.size.x), &(mat.size.y), &(mat.size.z), grayscale);
    mat.size.z = grayscale ? 1 : 3;
    allocated = true;
}

Image::~Image(){
    if(allocated)
        stbi_image_free(mat.data);
}

bool Image::save(std::string filename){
    return stbi_write_png(filename.c_str(), mat.size.x, mat.size.y, mat.size.z, mat.data, mat.size.x * mat.size.z);
}

void cpu_gaussian(Mat3D<uchar> in, Mat3D<uchar> out){
    if(in.size.x != out.size.x || in.size.y != out.size.y || in.size.z != out.size.z)
        throw std::runtime_error("Matrix sizes differ");

    int kerSize = 5;
    int gaussKernel[5][5] = {
        {1,4,6,4,1},
        {4,16,24,16,4},
        {6,24,36,24,6},
        {4,16,24,16,4},
        {1,4,6,4,1},
    };
    int kernelSum = 256;

    int3 pos;
    for(pos.x= 0; pos.x < in.size.x; pos.x++){
        for(pos.y = 0; pos.y < in.size.y; pos.y++){
            for(pos.z = 0; pos.z < in.size.z; pos.z++){
                int acum = 0;
                for(int i = 0; i < kerSize; i++){
                    for(int j = 0; j < kerSize; j++){
                        int3 kerPos;
                        kerPos.x = pos.x + i - (kerSize/2);
                        kerPos.y = pos.y + j - (kerSize/2);
                        kerPos.z = pos.z;
                        if( kerPos.x >= 0 && kerPos.x < in.size.x &&
                            kerPos.y >= 0 && kerPos.y < in.size.y ){
                            acum += (int) in[kerPos] * gaussKernel[i][j];
                        }   
                    }
                }
                acum /= kernelSum;
                out[pos] = acum;
            }
        }
    }
    
}