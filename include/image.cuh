#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <vector>

#include "matrix.cuh"

struct Image{
    Mat3D<uchar> mat;

    bool read(std::string filename);
    bool close();
    bool save(std::string filename);
};

#endif