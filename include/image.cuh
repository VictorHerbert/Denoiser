#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <string>

struct Image{
    int3 size;
    uchar* data;

    Image(std::string filename);
    bool save(std::string filename);
    ~Image();

    uchar getPx(int3 pos);
    uchar setPx(int3 pos, uchar col);
};

#endif