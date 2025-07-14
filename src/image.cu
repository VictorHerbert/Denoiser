#include "image.cuh"

#include <string>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

#include "third_party/helper_math.h"

#include <stdexcept>


Image::Image(float3* fmat, int2 shape){
    vBuffer.resize(totalSize(shape));
    for(int i = 0; i < vBuffer.size(); i+=3){
        buffer[i] = static_cast<uchar>(fmat[i].x*255);
        buffer[i+1] = static_cast<uchar>(fmat[i].y*255);
        buffer[i+2] = static_cast<uchar>(fmat[i].z*255);
    }
    buffer = vBuffer.data();
    stbi_allocated = false;
}


Image::Image(std::string filename){    
    buffer = (uchar*) stbi_load(filename.c_str(), &(shape.x), &(shape.y), &(shape.z), 0);
    stbi_allocated = true;
}

Image::~Image(){
    if(stbi_allocated)
        close();
}

bool Image::close(){
    stbi_image_free(buffer);
    return true;
}

bool Image::save(std::string filename){
    return stbi_write_png(filename.c_str(), shape.x, shape.y, shape.z, buffer, shape.x * shape.z);
}

std::vector<float3> fVecFromImage(const Image& img){
    std::vector<float3> out(img.shape.x * img.shape.y);
    for(int i = 0; i < totalSize(img.shape); i+=3)
        out[i/3] = {
            static_cast<float>(img.buffer[i])/255,
            static_cast<float>(img.buffer[i+1])/255,
            static_cast<float>(img.buffer[i+2])/255
        };

    return out;
}