#include "image.cuh"

#include <string>


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"


Image::Image(std::string filename){
    this->data = (uchar1*) stbi_load(filename.c_str(), &(this->size.x), &(this->size.y), &(this->size.z), 0);
}

Image::~Image(){
    stbi_image_free(this->data);
}

bool Image::save(std::string filename){
    return stbi_write_png(filename.c_str(), this->size.x, this->size.y, this->size.z, this->data, this->size.x * this->size.z);
}