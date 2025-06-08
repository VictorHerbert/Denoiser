#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <string>
#include <vector>

#include "cuda_runtime.h"
#include <stdexcept>

typedef unsigned char uchar;


template <typename T>
struct Mat3D {
    int3 size;
    T* data = nullptr;

    inline int totalSize(){
        return size.x * size.y * size.z;
    }

    inline int index(int x, int y, int z) const {
        return x * size.y * size.z + y * size.z + z;
    }

    inline T& operator()(int x, int y, int z) {
        return data[index(x, y, z)];
    }

    inline const T& operator()(int x, int y, int z) const {
        return data[index(x, y, z)];
    }
};


#endif