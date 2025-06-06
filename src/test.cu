#include "test.cuh"

#include <iostream>

#include "image.cuh"

bool test_conv(){
    uchar inData[5][5][1] = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 16, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    };
    uchar outData[5][5][1];
    Mat3D<uchar> in = {make_int3(5,5,1), &inData[0][0][0]};
    Mat3D<uchar> out = {make_int3(5,5,1), &outData[0][0][0]};

    cpu_gaussian(in, out);

    return true;
}


bool test_gauss(){
    Image input("sample/cornell/32/Render.png");
    Image output(input.mat.size);

    cpu_gaussian(input.mat, output.mat);

    output.save("build/sample/cornell32.png");

    return true;
}

void test(){
    std::cout << "Testing started\n" << std::endl;
    TEST_FUNC(test_conv);
    TEST_FUNC(test_gauss);
    std::cout << "\nFinished" << std::endl;   
}