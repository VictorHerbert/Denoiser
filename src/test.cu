#include "test.cuh"

#include <iostream>

#include "matrix.cuh"
#include "image.cuh"
#include "filter.cuh"

bool test_image(){    
    Image image;
    image.read("sample/cornell/32/Render.png");
    image.save("build/sample/test_image.png");

    return true;
}

bool test_gauss_filter(){
    Image input;
    input.read("sample/cornell/32/Render.png");
    
    std::vector<float> inBuffer(input.mat.totalSize());
    Mat3D<float> inFloat = {input.mat.size, inBuffer.data()};
    for(int i = 0; i < inFloat.totalSize(); i++)
        inFloat.data[i] = static_cast<float>(input.mat.data[i])/255;

    std::vector<float> outBuffer(input.mat.totalSize());
    Mat3D<float> outFloat = {input.mat.size, outBuffer.data()};
    
    
    gaussianFilterCPU(inFloat, outFloat);

    input.close();

    std::vector<uchar> outCharBuffer(input.mat.totalSize());
    Mat3D<uchar> outChar = {outFloat.size, outCharBuffer.data()};
    for(int i = 0; i < outFloat.totalSize(); i++)
        outChar.data[i] = static_cast<uchar>(outFloat.data[i]*255);
        
    Image output = {outChar};

    output.save("build/sample/gaussian.png");

    return true;
}

void test(){
    std::cout << "Testing started\n" << std::endl;
    TEST_FUNC(test_image);
    TEST_FUNC(test_gauss_filter);
    std::cout << "\nFinished" << std::endl;   
}