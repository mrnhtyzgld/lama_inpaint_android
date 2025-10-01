#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <nnapi_provider_factory.h>
#include <android/log.h>
#include <onnxruntime_session_options_config_keys.h>

// ----- Minimal ekler: NNAPI/XNNPACK ve genel runner ayarları -----

struct NnapiOptions {
    enum class Flag : uint32_t {
        None        = 0,
        UseFp16     = 1u << 0,  // NNAPI_FLAG_USE_FP16
        CpuDisabled = 1u << 1,  // NNAPI_FLAG_CPU_DISABLED
        CpuOnly     = 1u << 2,  // NNAPI_FLAG_CPU_ONLY
        UseNchw     = 1u << 3,  // NNAPI_FLAG_USE_NCHW
    };
    static constexpr uint32_t to_raw(Flag f) { return static_cast<uint32_t>(f); }

    // bitwise helpers
    friend constexpr Flag operator|(Flag a, Flag b) {
        return static_cast<Flag>( static_cast<uint32_t>(a) | static_cast<uint32_t>(b) );
    }
    friend constexpr Flag& operator|=(Flag& a, Flag b) { a = a | b; return a; }

    Flag flags = Flag::None;  // default: tüm bayraklar kapalı
};

struct RunnerSettings {
    int  num_cpu_cores;

    bool use_xnnpack   = true;
    bool use_nnapi     = true;

    bool use_parallel_execution  = false; // github says parallel execution is deprecated but
    bool use_layout_optimization_instead_of_extended = false; // website said if not nnapi use extended but this seems faster

    NnapiOptions   nnapi{};
};

class InferenceRunner {
public:
    InferenceRunner() = default;

    // Provide the model path and configure internal settings
    void init_model(std::string model_path);

    // Execute inference with an image and its mask; returns output matrices
    std::vector<cv::Mat> run(const cv::Mat &image, const cv::Mat &mask);

    std::vector<uint8_t> runByteToByte(const std::vector<uint8_t> &imageBytes,
                                       const std::vector<uint8_t> &maskBytes);

private:
    // Helper functions
    void find_input_output_info_();

    cv::Mat ort_output_to_mat(const Ort::Value &out);

    void start_environment_(const RunnerSettings& s);

    cv::Mat decodeBytesToMat_(const std::vector<uint8_t> &bytes, int flags);

    std::vector<uint8_t> encodeMat_(const cv::Mat &img, const std::string &ext);

    std::string model_path_;
    int image_idx_;
    int mask_idx_;
    int image_width_, image_height_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    // ONNX Runtime objects
    Ort::Session session_{nullptr};
    Ort::MemoryInfo mem_info_{nullptr};
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
};
