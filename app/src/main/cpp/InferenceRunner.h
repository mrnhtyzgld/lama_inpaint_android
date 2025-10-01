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

    void start_environment_(int num_inter_threads, int num_intra_threads,
                                             GraphOptimizationLevel optimization_level,
                                             int num_cpu_core, bool use_xnn, bool use_nnapi);

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
