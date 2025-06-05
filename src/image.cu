#include "image.cuh"

#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"


void Framebuffer::save(std::string filename){
    stbi_write_png(filename.c_str(), this->size.x, this->size.y, this->channels, this->data, this->size.x * this->channels);
}