#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <memory>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <nnapi_provider_factory.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "config.h"

class ModelSession {
public:
    ModelSession(Ort::Env& env,
                 Ort::MemoryInfo& mem_info,
                 RunnerSettings s,
                 std::string model_path);

    std::vector<uint8_t> runEndToEnd(const std::vector<uint8_t>& imageBytes,
                                     const std::vector<uint8_t>& maskBytes);

    std::vector<cv::Mat> run(const cv::Mat& image, const cv::Mat& mask);

    const std::string& model_path() const { return model_path_; }

private:
    // SessionOptions
    Ort::SessionOptions init_session(RunnerSettings s);

    // Yardımcılar (CPP'de tanımlı)
    cv::Mat              decodeBytesToMat_(const std::vector<uint8_t>& bytes, int flags);
    std::vector<uint8_t> encodeMat_(const cv::Mat& img, const std::string& ext);
    void                 find_input_output_info_();
    cv::Mat              ort_output_to_mat(const Ort::Value& out);

private:
    // ORT
    Ort::Session     session_{nullptr};
    Ort::MemoryInfo& mem_info_;

    // Model & ayarlar
    std::string    model_path_;
    RunnerSettings settings_;

    // IO bilgileri
    int image_idx_;
    int mask_idx_;
    int image_width_;
    int image_height_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};
