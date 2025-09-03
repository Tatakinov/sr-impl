#include <cmath>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <sstream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>
#include <vector>

int main() {
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session(nullptr);
    try {
        session = Ort::Session(env, "model.onnx", session_options);
    }
    catch (Ort::Exception &e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    std::array<int64_t, 4> input_shape = {1, 1, 0, 0};
    std::array<int64_t, 4> output_shape = {1, 1, 0, 0};
    int w, h, _bpp;
    unsigned char *p = stbi_load("input.png", &w, &h, &_bpp, 4);
    if (p == nullptr) {
        return 1;
    }
    input_shape[2] = h;
    input_shape[3] = w;
    output_shape[2] = 2 * h;
    output_shape[3] = 2 * w;

    std::vector<float> input;
    input.resize(1 * w * h);
    std::vector<float> output;
    output.resize(1 * 4 * w * h);
    std::vector<unsigned char> x2;
    x2.resize(4 * w * h * 4);

    for (int c = 0; c < 4; c++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = (y * w + x) * 4;
                input[y * w + x] = p[i + c] / 255.0;
            }
        }
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto input_tensor = Ort::Value::CreateTensor<float>(mem_info, input.data(), input.size(), input_shape.data(), input_shape.size());
        auto output_tensor = Ort::Value::CreateTensor<float>(mem_info, output.data(), output.size(), output_shape.data(), output_shape.size());
        const char *input_name[] = { "input" };
        const char *output_name[] = { "output" };
        Ort::RunOptions run_options;
        try {
            session.Run(run_options, input_name, &input_tensor, 1, output_name, &output_tensor, 1);
        }
        catch (Ort::Exception &e) {
            std::cout << e.what() << std::endl;
            return 1;
        }
        for (int y = 0; y < 2 * h; y++) {
            for (int x = 0; x < 2 * w; x++) {
                int index = (y * (2 * w) + x);
                int byte = std::round(output[index] * 255);
                byte = std::max(0, std::min(255, byte));
                index *= 4;
                x2[index + c] = byte;
            }
        }
    }

    stbi_image_free(p);

    stbi_write_png("output.png", 2 * w, 2 * h, 4, x2.data(), 0);
    return 0;
}
