#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <string>

struct Framebuffer{
    uint2 size;
    uchar1* data;
    const int channels = 3;

    Framebuffer(std::string filename);
    void save(std::string filename);
    uchar1 getPx(int3 pos);
    uchar1 setPx(int3 pos, uchar1 col);
};

#endif