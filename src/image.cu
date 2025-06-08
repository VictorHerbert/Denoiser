#include "image.cuh"

#include <string>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

#include "third_party/helper_math.h"

#include <stdexcept>


bool Image::read(std::string filename){
    mat.data = (uchar*) stbi_load(filename.c_str(), &(mat.size.x), &(mat.size.y), &(mat.size.z), 0);
    return true;
}

bool Image::close(){
    stbi_image_free(mat.data);
    return true;
}

bool Image::save(std::string filename){
    return stbi_write_png(filename.c_str(), mat.size.x, mat.size.y, mat.size.z, mat.data, mat.size.x * mat.size.z);
}
