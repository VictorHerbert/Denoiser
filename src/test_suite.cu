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

TEST(waveletfilter){
    Image render("render/cornell/render_1.png");
    Image golden("render/cornell/render_8192.png");

    auto golden_f = fmatFromImage(golden);
    auto render_f = fmatFromImage(render);
    CPUMat3D<float> out_f(render.mat.size);

    waveletfilterCPU(render_f, out_f,
        fmatFromImage(Image("render/cornell/albedo.png")),
        fmatFromImage(Image("render/cornell/normal.png")),
        5, 1, 1e1, 1, 1e-2);

    float3 preSnr = snrCPU(golden_f, render_f);
    float3 posSnr = snrCPU(golden_f, out_f);

    Image output(out_f);
    output.save("build/test/crossBilateralfilterCPU.png");

    printf("\nSNR %f %f %f -> %f %f %f\n", preSnr.x, preSnr.y, preSnr.z, posSnr.x, posSnr.y, posSnr.z);

    return TestStatus::SUCCESS;
}