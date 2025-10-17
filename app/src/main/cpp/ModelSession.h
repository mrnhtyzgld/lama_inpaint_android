#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class ModelSession {
public:
    ModelSession(std::string model_path);

    std::vector<uint8_t> runEndToEnd(const std::vector<uint8_t> &imageBytes,
                                     const std::vector<uint8_t> &maskBytes);

    std::vector<cv::Mat> run(const cv::Mat &image, const cv::Mat &mask);

    const std::string &model_path() const { return model_path_; }

private:
    cv::Mat decodeBytesToMat_(const std::vector<uint8_t> &bytes, int flags);

    std::vector<uint8_t> encodeMat_(const cv::Mat &img, const std::string &ext);

    cv::dnn::Net net;

    // Model & settings
    std::string model_path_;

    // IO info
    int image_width_;
    int image_height_;

    std::vector<std::vector<int64_t>> input_shapes_, output_shapes_;
    size_t in_count, out_count;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};
