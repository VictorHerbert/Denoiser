#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <string>
#include <vector>

#include "cuda_runtime.h"
#include <stdexcept>

typedef unsigned char uchar;

// Equality operator
inline bool operator==(const int3& a, const int3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// Inequality operator
inline bool operator!=(const int3& a, const int3& b) {
    return !(a == b);
}

template <typename T>
struct Mat3D {
    int3 size;
    T* data = nullptr;

    inline int totalSize() const {
        return size.x * size.y * size.z;
    }

    inline int index(int x, int y, int z) const {
        return x * size.y * size.z + y * size.z + z;
    }

    inline int index(uint3 pos) const {
        return index(pos.x, pos.y, pos.z);
    }

    inline T& operator()(int x, int y, int z) {
        return data[index(x, y, z)];
    }

    inline const T& operator()(int x, int y, int z) const {
        return data[index(x, y, z)];
    }

    inline T& operator()(uint3 pos) {
        return data[index(pos)];
    }

    inline const T& operator()(uint3 pos) const {
        return data[index(pos)];
    }

    inline T& operator[](uint3 pos) {
        return data[index(pos)];
    }

    inline const T& operator[](uint3 pos) const {
        return data[index(pos)];
    }

    inline bool advanceIterator(uint3& pos) {
        pos.z++;
        if (pos.z < size.z)
            return true;

        pos.z = 0;
        pos.y++;
        if (pos.y < size.y)
            return true;

        pos.y = 0;
        pos.x++;
        if (pos.x < size.x)
            return true;

        return false;
    }

};


template <typename T>
struct CPUMat3D : Mat3D<T> {
    std::vector<T> vData;

    CPUMat3D(){}
    CPUMat3D(int3 size){
        this->size = size;
        vData.resize(this->totalSize());
        this->data = vData.data();
    }
};




#endif