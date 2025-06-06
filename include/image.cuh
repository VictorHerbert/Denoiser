#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <string>
#include <vector>

#include "cuda_runtime.h"
#include <stdexcept>

typedef unsigned char uchar;

template <typename T>
struct Mat3D{
    int3 size;
    T* data;

    // Convert 3D index to linear index
    inline int index(int3 idx) const {
        if (idx.x < 0 || idx.x >= size.x ||
            idx.y < 0 || idx.y >= size.y ||
            idx.z < 0 || idx.z >= size.z) {
            throw std::out_of_range("Index out of bounds in Mat3D");
        }
        return idx.x * size.y * size.z + idx.y * size.z + idx.z;
    }

    T& operator[](int3 idx) {
        return data[index(idx)];
    }

    const T& operator[](int3 idx) const {
        return data[index(idx)];
    }
};

template <typename T>
struct Mat2D{
    int2 size;
    T* data;
};

struct Image{
    Mat3D<uchar> mat;
    bool allocated;
    std::vector<uchar> vData;

    Image(int3 size);
    Image(std::string filename, bool grayscale = false);
    bool save(std::string filename);
    ~Image();    
};

void cpu_gaussian(Mat3D<uchar> in, Mat3D<uchar> out);

#endif