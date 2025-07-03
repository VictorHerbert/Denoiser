#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <vector>

#include "matrix.cuh"

struct Image{
    CPUMat3D<uchar> mat;
    bool stbi_allocated = false;
    
    Image(){}
    Image(CPUMat3D<float> fmat){
        mat = CPUMat3D<uchar>(fmat.size);
        for(int i = 0; i < fmat.totalSize(); i++)
            mat.data[i] = static_cast<uchar>(fmat.data[i]*255);
    }
    
    bool read(std::string filename);
    bool close();
    bool save(std::string filename);
};

CPUMat3D<float> fmatFromImage(Image img);

#endif