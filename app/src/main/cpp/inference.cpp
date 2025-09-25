#include "inference.h"
#include "InferenceRunner.h"
#include "onnxruntime_cxx_api.h"
#include <cassert>
#include <android/log.h>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace {

    // NCHW (N=1) float tensor -> interleaved uint8 RGB
    inline std::vector<uint8_t> ort_output_to_buffer_rgb8(const Ort::Value &out) {
        auto info = out.GetTensorTypeAndShapeInfo();
        auto shp = info.GetShape(); // expected NCHW
        if (shp.size() != 4 || shp[0] != 1) {
            throw std::runtime_error("Expected NCHW with N=1.");
        }
        const int64_t C = shp[1], H = shp[2], W = shp[3];
        if (C != 1 && C != 3) {
            throw std::runtime_error("Only C=1 or C=3 supported.");
        }
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("Only float32 output is supported here.");
        }

        const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);
        const float *ptr = out.GetTensorData<float>();

        // always 3 channel output
        constexpr int outC = 3;
        std::vector<uint8_t> data(static_cast<size_t>(W * H * outC));

        auto clamp_to_u8 = [&](float v) -> uint8_t {
            // expected [0,255]
            if (v < 0.f) v = 0.f;
            if (v > 255.f) v = 255.f;
            return static_cast<uint8_t>(v + 0.5f);
        };

        if (C == 1) {
            // CHW (single channel) -> expand to HWC RGB
            for (int64_t y = 0; y < H; ++y) {
                for (int64_t x = 0; x < W; ++x) {
                    size_t src = static_cast<size_t>(y * W + x); // 0*H*W + y*W + x
                    uint8_t g = clamp_to_u8(ptr[src]);
                    size_t dst = static_cast<size_t>((y * W + x) * outC);
                    data[dst + 0] = g; // R
                    data[dst + 1] = g; // G
                    data[dst + 2] = g; // B
                }
            }
        } else { // C == 3
            const float *c0 = ptr + plane * 0; // R
            const float *c1 = ptr + plane * 1; // G
            const float *c2 = ptr + plane * 2; // B
            for (int64_t y = 0; y < H; ++y) {
                for (int64_t x = 0; x < W; ++x) {
                    size_t idx = static_cast<size_t>(y * W + x);
                    uint8_t r = clamp_to_u8(c0[idx]);
                    uint8_t g = clamp_to_u8(c1[idx]);
                    uint8_t b = clamp_to_u8(c2[idx]);
                    size_t dst = static_cast<size_t>((y * W + x) * outC);
                    data[dst + 0] = r;
                    data[dst + 1] = g;
                    data[dst + 2] = b;
                }
            }
        }

        return data;
    }

} //namespace

namespace inference {

    std::vector<float> infer(SessionCache *session_cache,
                                float *image_data,
                                float *mask_data,
                                int64_t batch_size,
                                int64_t image_channels,
                                int64_t image_rows,
                                int64_t image_cols) {
        // I/O names
        std::vector<const char *> input_names = {"image", "mask"};
        std::vector<const char *> output_names = {"output"};

        const size_t input_count = input_names.size();
        const size_t output_count = output_names.size();

        // Data Shapes
        std::vector<int64_t> image_shape{batch_size, image_channels, image_rows,
                                         image_cols}; // 1x3xHxW
        std::vector<int64_t> mask_shape{batch_size, 1, image_rows, image_cols}; // 1x1xHxW

        // Input Tensors
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                                                 OrtMemTypeDefault);

        std::vector<Ort::Value> input_values;
        input_values.reserve(2);

        const size_t img_bytes = static_cast<size_t>(batch_size) *
                                 static_cast<size_t>(image_channels) *
                                 static_cast<size_t>(image_rows) *
                                 static_cast<size_t>(image_cols) * sizeof(float);

        const size_t msk_bytes = static_cast<size_t>(batch_size) *
                                 /*mask_channels=*/1u *
                                 static_cast<size_t>(image_rows) *
                                 static_cast<size_t>(image_cols) * sizeof(float);

        input_values.emplace_back(Ort::Value::CreateTensor(
                memory_info, image_data, img_bytes,
                image_shape.data(), image_shape.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        input_values.emplace_back(Ort::Value::CreateTensor(
                memory_info, mask_data, msk_bytes,
                mask_shape.data(), mask_shape.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Outputs
        std::vector<Ort::Value> output_values =
                session_cache->inference_session->Run(
                        Ort::RunOptions{},                      // or Ort::RunOptions()
                        input_names.data(), input_values.data(), input_count,
                        output_names.data(),                    // output names
                        output_count
                );

        // Convert output to RGB8 and return as float [0..255] (use existing helper)
        std::vector<float> result;
        for (auto &s: output_values) {
            auto rgb = ort_output_to_buffer_rgb8(s);             // W*H*3 bytes
            result.reserve(result.size() + rgb.size());
            for (uint8_t b: rgb) result.push_back(static_cast<float>(b));
        }
        return result;
    }

} // namespace inference
