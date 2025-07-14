#include "test.cuh"

#include <iostream>

#include "matrix.cuh"
#include "image.cuh"
#include "filter.cuh"

SKIP(image){
    Image image("render/cornell/render_1.png");
    image.save("build/test/render_1.png");
    return TestStatus::SUCCESS;
}

SKIP(snr){
    Image original("render/cornell/render_1.png");
    Image noisy("render/cornell/render_8192.png");
    int2 shape = {original.shape.x, original.shape.y};

    auto originalFmat = fVecFromImage(original);
    auto noisyFmat = fVecFromImage(noisy);

    float3 zeroSnr = snrCPU(originalFmat.data(), originalFmat.data(), shape);
    float3 snr = snrCPU(originalFmat.data(), noisyFmat.data(), shape);

    return TestStatus::SUCCESS;
}

TEST(waveletfilter){
    Image render("render/cornell/render_1.png");
    Image golden("render/cornell/render_8192.png");
    int2 shape = {render.shape.x, render.shape.y};

    auto golden_f = fVecFromImage(golden);
    auto render_f = fVecFromImage(render);
    std::vector<float3> out_f(totalSize(shape));

    auto fAlbedo = fVecFromImage(Image("render/cornell/albedo.png"));
    auto fNormal = fVecFromImage(Image("render/cornell/normal.png"));

    waveletfilterCPU(render_f.data(), out_f.data(),
        fAlbedo.data(),
        fNormal.data(),
        shape,
        5, 1, 1, 1, 1, 1e-2);

    float3 preSnr = snrCPU(golden_f.data(), render_f.data(), shape);
    float3 posSnr = snrCPU(golden_f.data(), out_f.data(), shape);

    Image output(out_f.data(), shape);
    output.save("build/test/crossBilateralfilterCPU.png");

    printf("\nSNR %f %f %f -> %f %f %f\n", preSnr.x, preSnr.y, preSnr.z, posSnr.x, posSnr.y, posSnr.z);

    return TestStatus::SUCCESS;
}