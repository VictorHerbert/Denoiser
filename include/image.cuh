#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <vector>

#include "matrix.cuh"

struct Image{
    CPUMat3D<uchar> mat;
    bool stbi_allocated = false;
    
    Image(){}
    Image(std::string filename);
    Image(CPUMat3D<float> fmat);
    ~Image();
    
    bool close();
    bool save(std::string filename);
};

CPUMat3D<float> fmatFromImage(const Image& img);

#endif