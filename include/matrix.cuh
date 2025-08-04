#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <string>
#include <vector>
#include <stdio.h>

#include "cuda_runtime.h"
#include <stdexcept>

typedef unsigned char uchar;

inline bool operator==(const int3& a, const int3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline bool operator!=(const int3& a, const int3& b) {
    return !(a == b);
}

template <typename T>
struct CudaVector{
    T* data;
    size_t size;

    CudaVector(size_t size) : size(size) {
        cudaMalloc(&data, size * sizeof(T));
    }

    CudaVector(T* v, size_t size) : size(size) {
        cudaMalloc(&data, size * sizeof(T));
        cudaMemcpy(data, v, size * sizeof(T), cudaMemcpyHostToDevice);
    }

    ~CudaVector() {
        cudaFree(data);
    }

};

__device__ __host__ int index(int x, int y, int2 size);
__device__ __host__ int index(int2 p, int2 size);
bool advanceIterator(int2& pos, int2 size);
__device__ __host__ int totalSize(int2 shape);
__device__ __host__ int totalSize(int3 shape);

#endif