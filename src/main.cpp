#include <iostream>

#include "matrix.cuh"
#include "third_party/helper_math.h"

#include "test.cuh"

int main(int argc, char* argv[]){
    test();
    return 0;

    
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

    /*if(!output.save(savePath)){
        std::cerr << "Couldnt write to " << savePath << "\n";
        return 1;
    }*/

    return 0;
}