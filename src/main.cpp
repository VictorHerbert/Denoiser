#include <iostream>

#include "image.cuh"
#include "third_party/helper_math.h"

int main(int argc, char* argv[]){
    std::string renderPath;
    std::string savePath;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-r" && i + 1 < argc) {
            renderPath = argv[i + 1];
            i++;
        } else if (arg == "-s" && i + 1 < argc) {
            savePath = argv[i + 1];
            i++;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            return 1;
        }
    }

    Image image(renderPath);

    int3 pos;
    for(pos.x = 0; pos.x < image.size.x; pos.x++)
        for(pos.y = 0; pos.y < image.size.y; pos.y++)
            for(pos.z = 0; pos.z < image.size.z; pos.z++)
                image.setPx(pos, 255 - image.getPx(pos));


    if(!image.save(savePath)){
        std::cerr << "Couldnt write to " << savePath << "\n";
        return 1;
    }

    return 0;
}