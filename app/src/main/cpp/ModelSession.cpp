#include "ModelSession.h"
#include "logging.h"
#include <opencv2/dnn.hpp>

using namespace cv::dnn;

/*
 * https://github.com/devingarg/onnx-quantization/blob/main/resnet_inference.cpp
 */

ModelSession::ModelSession(std::string model_path) {
    model_path_ = model_path;
    net = readNetFromONNX(model_path_);
    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(DNN_TARGET_CPU);
}


std::vector<uint8_t> ModelSession::runEndToEnd(const std::vector<uint8_t> &imageBytes,
                                               const std::vector<uint8_t> &maskBytes) {
    if (imageBytes.empty())
        throw std::invalid_argument("runEndToEnd: imageBytes is empty");
    if (maskBytes.empty())
        throw std::invalid_argument("runEndToEnd: maskBytes is empty");
    cv::Mat image = decodeBytesToMat_(imageBytes, cv::IMREAD_COLOR);     // BGR, 3ch
    cv::Mat mask = decodeBytesToMat_(maskBytes, cv::IMREAD_GRAYSCALE); // 1ch

    auto outputMats = run(image, mask);
    if (outputMats.empty()) throw std::runtime_error("no outputs from session");
    //Take first input
    return encodeMat_(outputMats[0], ".png");
}

std::vector<cv::Mat> ModelSession::run(const cv::Mat &image, const cv::Mat &mask) {
    try {


        // Inputs
        if (image.empty())
            throw std::runtime_error("image is empty");
        if (mask.empty())
            throw std::runtime_error("mask is empty");
        if (image.channels() != 3)
            throw std::runtime_error("image must have 3 channels (BGR)");
        if (mask.channels() != 1)
            throw std::runtime_error("mask must have 1 channel (grayscale)");

        cv::Size target(image_width_, image_height_);

        cv::Mat mat_image = image;
        cv::Mat mat_mask = mask;

        if (mat_mask.size() != target) {
            cv::resize(mat_mask, mat_mask, target, 0, 0, cv::INTER_NEAREST);
        }
        cv::threshold(mat_mask, mat_mask, 127, 255, cv::THRESH_BINARY);

        mat_image = cv::dnn::blobFromImage(
                mat_image, 1.f / 255.f, target, cv::Scalar(), /*swapRB*/
                true, /*crop*/ false, CV_32F);
        mat_mask = cv::dnn::blobFromImage(
                mat_mask, 1.f / 255.f, target, cv::Scalar(), /*swapRB*/
                false, /*crop*/ false, CV_32F);
        /*

        // Outputs
        std::vector<cv::Mat> outputs;
        try {
            outputs = session_.Run(Ort::RunOptions{},
                                   input_names_c.data(), inputs.data(), inputs.size(),
                                   output_names_c.data(), output_names_c.size());
        } catch (const Ort::Exception &e) {
            __android_log_print(ANDROID_LOG_ERROR, "cpponnxrunner",
                                "session.Run Ort::Exception: %s", e.what());
            throw;
        } catch (const std::exception &e) {
            __android_log_print(ANDROID_LOG_ERROR, "cpponnxrunner",
                                "session.Run std::exception: %s", e.what());
            throw;
        } catch (...) {
            __android_log_print(ANDROID_LOG_ERROR, "cpponnxrunner",
                                "session.Run unknown exception");
            throw;
        }



        // Process outputs
        std::vector<cv::Mat> output_mats(outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i)
            output_mats[i] = ort_output_to_mat(outputs[i]);

        __android_log_print(ANDROID_LOG_INFO, "cpponnxrunner",
                            "run(): img[%dx%d ch=%d type=%d] mask[%dx%d ch=%d type=%d] target=%dx%d | in=%zu out=%zu",
                            image.cols, image.rows, image.channels(), image.type(),
                            mask.cols, mask.rows, mask.channels(), mask.type(),
                            image_width_, image_height_,
                            input_names_.size(), output_names_.size());
        */
        std::vector<cv::Mat> output_mats(1);
        return output_mats;
    } catch (const std::exception &e) {
        __android_log_print(ANDROID_LOG_ERROR, "cpponnxrunner",
                            "runEndToEnd std::exception: %s", e.what());
        throw;
    } catch (...) {
        __android_log_print(ANDROID_LOG_ERROR, "cpponnxrunner",
                            "runEndToEnd unknown exception");
        throw;
    }
}

cv::Mat ModelSession::decodeBytesToMat_(const std::vector<uint8_t> &bytes, int flags) {
    if (bytes.empty()) throw std::runtime_error("decodeBytesToMat_: empty buffer");
    cv::Mat buf(1, static_cast<int>(bytes.size()), CV_8U, const_cast<uint8_t *>(bytes.data()));
    cv::Mat img = cv::imdecode(buf, flags);
    if (img.empty()) throw std::runtime_error("imdecode failed");
    return img;
}

std::vector<uint8_t> ModelSession::encodeMat_(const cv::Mat &img, const std::string &ext = ".png") {
    std::vector<uint8_t> out;
    std::vector<int> params;
    if (ext == ".jpg" || ext == ".jpeg") {
        params = {cv::IMWRITE_JPEG_QUALITY, 100};
    }
    if (!cv::imencode(ext, img, out, params)) {
        throw std::runtime_error("imencode failed");
    }
    return out;
}
