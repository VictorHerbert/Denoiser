#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <string>
#include <vector>

#include "cuda_runtime.h"
#include <stdexcept>

typedef unsigned char uchar;

inline bool operator==(const int3& a, const int3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline bool operator!=(const int3& a, const int3& b) {
    return !(a == b);
}

int index(int x, int y, int2 size);
int index(int2 p, int2 size);
bool advanceIterator(int2& pos, int2 size);
int totalSize(int2 shape);
int totalSize(int3 shape);

#endif