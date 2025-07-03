#include "test.cuh"

#include <iostream>

#include "matrix.cuh"
#include "image.cuh"
#include "filter.cuh"

TEST(image){
    Image image("render/cornell/render_1.png");
    image.save("build/test/render_1.png");
    return TestStatus::SUCCESS;
}

TEST(snr){
    Image original("render/cornell/render_1.png");
    Image noisy("render/cornell/render_8192.png");

    auto originalFmat = fmatFromImage(original);
    auto noisyFmat = fmatFromImage(noisy);

    float3 zeroSnr = snrCPU(originalFmat, originalFmat);
    //float3 snr = snrCPU(originalFmat, noisyFmat);
    
    return TestStatus::SUCCESS;
}

TEST(crossBilateralfilter){
    Image input("render/cornell/render_32.png");
    Image golden("render/cornell/render_8192.png");
    auto fgolden = fmatFromImage(golden);

    CPUMat3D<float> inFloat = fmatFromImage(input);
    CPUMat3D<float> outFloat(input.mat.size);
    
    crossBilateralfilterCPU(inFloat, outFloat, inFloat, 7, 2, 0.5, 2.0);

    float3 preSnr = snrCPU(fgolden, inFloat);
    float3 posSnr = snrCPU(fgolden, outFloat);
        
    Image output(outFloat);
    output.save("build/test/crossBilateralfilterCPU.png");    

    return TestStatus::SUCCESS;
}
