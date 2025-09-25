#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

class InferenceRunner
{
public:
    InferenceRunner() = default;

    // Provide the model path and configure internal settings
    void init_model(std::string model_path);

    // Execute inference with an image and its mask; returns output matrices
    std::vector<cv::Mat> run(const cv::Mat &image, const cv::Mat &mask);

private:
    // Helper functions
    void find_input_output_info_();
    cv::Mat ort_output_to_mat(const Ort::Value &out);
    void start_environment_();

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
