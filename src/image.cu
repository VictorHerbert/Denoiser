#include "image.cuh"

#include <string>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

#include "third_party/helper_math.h"

#include <stdexcept>


Image::Image(CPUMat3D<float> fmat){
    mat = CPUMat3D<uchar>(fmat.size);
    for(int i = 0; i < fmat.totalSize(); i++)
        mat.data[i] = static_cast<uchar>(fmat.data[i]*255);
    stbi_allocated = false;
}


Image::Image(std::string filename){
    mat.data = (uchar*) stbi_load(filename.c_str(), &(mat.size.x), &(mat.size.y), &(mat.size.z), 0);
    stbi_allocated = true;
}

Image::~Image(){
    if(stbi_allocated)
        close();
}

bool Image::close(){
    stbi_image_free(mat.data);
    return true;
}

bool Image::save(std::string filename){
    return stbi_write_png(filename.c_str(), mat.size.x, mat.size.y, mat.size.z, mat.data, mat.size.x * mat.size.z);
}

CPUMat3D<float> fmatFromImage(const Image& img){
    CPUMat3D<float> out(img.mat.size);
    for(int i = 0; i < img.mat.totalSize(); i++)
        out.data[i] = static_cast<float>(img.mat.data[i])/255;

    return out;
}