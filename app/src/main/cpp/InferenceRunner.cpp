#include "InferenceRunner.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <stdexcept>
#include <nnapi_provider_factory.h>
#include <android/log.h>
#include <onnxruntime_session_options_config_keys.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <stdexcept>

void InferenceRunner::init_model(std::string model_path) {
    model_path_ = model_path;
    find_input_output_info_();
    start_environment_(1,
                       1,
                       GraphOptimizationLevel::ORT_ENABLE_ALL,
                       "CPUExecutionProvider");


}

void InferenceRunner::start_environment_(int num_inter_threads, int num_intra_threads,
                                         GraphOptimizationLevel optimization_level,
                                         std::string provider_ = "") {
    if (session_) return;
    // TODO make use of different providers
    auto provider = Ort::GetAvailableProviders().front();

    // Setting up ONNX environment
    mem_info_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    env_ = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "Default");

    sessionOptions_.SetInterOpNumThreads(num_inter_threads);
    sessionOptions_.SetIntraOpNumThreads(num_intra_threads);
    // optimization will take time and memory during startup
    sessionOptions_.SetGraphOptimizationLevel(optimization_level);

    // 2) Derlemeye gömülü EP’leri logla
    {
        auto provs = Ort::GetAvailableProviders();
        for (auto &p: provs) LOGI("Available EP: %s", p.c_str());
    }


    // NNAPI
    //uint32_t nnapi_flags = 0;
    //nnapi_flags |= NNAPI_FLAG_USE_FP16;
    //nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
    //nnapi_flags |= NNAPI_FLAG_CPU_ONLY;
    //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sessionOptions_, nnapi_flags));

    sessionOptions_.EnableProfiling("onnx_profile.json");


    // Start an ONNX Runtime session and create CPU memory info for input tensors.
    // model path is const wchar_t*
    const ORTCHAR_T *kModelPath = model_path_.c_str();
    session_ = Ort::Session(env_, kModelPath, sessionOptions_);
}

std::vector<uint8_t> InferenceRunner::runByteToByte(const std::vector<uint8_t> &imageBytes,
                                                    const std::vector<uint8_t> &maskBytes) {
    if (imageBytes.empty())
        throw std::invalid_argument("runByteToByte: imageBytes is empty");
    if (maskBytes.empty())
        throw std::invalid_argument("runByteToByte: maskBytes is empty");
    cv::Mat image = decodeBytesToMat_(imageBytes, cv::IMREAD_COLOR);     // BGR, 3ch
    cv::Mat mask = decodeBytesToMat_(maskBytes, cv::IMREAD_GRAYSCALE); // 1ch

    auto outputMats = run(image, mask);
    if (outputMats.empty()) throw std::runtime_error("no outputs from session");
    //Take first input
    return encodeMat_(outputMats[0], ".png");
}

std::vector<cv::Mat> InferenceRunner::run(const cv::Mat &image, const cv::Mat &mask) {
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

    mat_image = cv::dnn::blobFromImage(
            mat_image, 1.f / 255.f, target, cv::Scalar(), /*swapRB*/
            true, /*crop*/ false, CV_32F);
    mat_mask = cv::dnn::blobFromImage(
            mat_mask, 1.f / 255.f, target, cv::Scalar(), /*swapRB*/
            false, /*crop*/ false, CV_32F);

    auto *image_data = reinterpret_cast<float *>(mat_image.data);
    auto *mask_data = reinterpret_cast<float *>(mat_mask.data);
    if (mat_image.dims != 4 || mat_mask.dims != 4)
        throw std::runtime_error("blob must be 4D (NCHW).");

    std::vector<int64_t> image_shape = {mat_image.size[0], mat_image.size[1], mat_image.size[2],
                                        mat_image.size[3]}; // 1x3xHxW
    std::vector<int64_t> mask_shape = {mat_mask.size[0], mat_mask.size[1], mat_mask.size[2],
                                       mat_mask.size[3]};      // 1x1xHxW

    std::vector<const char *> input_names_c;
    for (auto &s: input_names_)
        input_names_c.push_back(s.c_str());

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
            mem_info_, image_data, (size_t) mat_image.total(),
            image_shape.data(), image_shape.size()));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
            mem_info_, mask_data, (size_t) mat_mask.total(),
            mask_shape.data(), mask_shape.size()));

    // Outputs
    std::vector<const char *> output_names_c;
    for (auto &s: output_names_)
        output_names_c.push_back(s.c_str());

    // Outputs
    auto outputs =
            session_.Run(Ort::RunOptions{},
                         input_names_c.data(), inputs.data(), inputs.size(),
                         output_names_c.data(), output_names_c.size()
            );


    // Process outputs
    std::vector<cv::Mat> output_mats(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i)
        output_mats[i] = ort_output_to_mat(outputs[i]);

    return output_mats;
}

cv::Mat InferenceRunner::decodeBytesToMat_(const std::vector<uint8_t> &bytes, int flags) {
    if (bytes.empty()) throw std::runtime_error("decodeBytesToMat_: empty buffer");
    cv::Mat buf(1, static_cast<int>(bytes.size()), CV_8U, const_cast<uint8_t *>(bytes.data()));
    cv::Mat img = cv::imdecode(buf, flags);
    if (img.empty()) throw std::runtime_error("imdecode failed");
    return img;
}

std::vector<uint8_t>
InferenceRunner::encodeMat_(const cv::Mat &img, const std::string &ext = ".png") {
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

void InferenceRunner::find_input_output_info_() {
    image_width_ = 512;
    image_height_ = 512;
    image_idx_ = 0;
    mask_idx_ = 1;
    input_names_ = {"image", "mask"};
    output_names_ = {"output"};
}

cv::Mat InferenceRunner::ort_output_to_mat(const Ort::Value &out) {
    // Take shape
    auto info = out.GetTensorTypeAndShapeInfo();
    auto shp = info.GetShape(); // NCHW expected)
    if (shp.size() != 4 || shp[0] != 1)
        throw std::runtime_error("Expected NCHW with N=1.");
    const int64_t C = shp[1], H = shp[2], W = shp[3];
    if (C != 1 && C != 3)
        throw std::runtime_error("Only C=1 or C=3 supported.");

    cv::Mat image_u8; // (CV_8U, 1 or 3 channel)
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);

    const auto *ptr = out.GetTensorData<float>();

    // CHW -> HWC (float)
    if (C == 1) {
        cv::Mat ch(H, W, CV_32F, const_cast<float *>(ptr));
        cv::Mat m = ch.clone();
        m.convertTo(image_u8, CV_8U);
    } else if (C == 3) {
        cv::Mat c0(H, W, CV_32F, const_cast<float *>(ptr + plane * 0));
        cv::Mat c1(H, W, CV_32F, const_cast<float *>(ptr + plane * 1));
        cv::Mat c2(H, W, CV_32F, const_cast<float *>(ptr + plane * 2));
        // Combine planar CHW channels into one interleaved HWC (RGB) image
        std::vector<cv::Mat> ch = {c0, c1, c2};
        cv::Mat img32f;
        cv::merge(ch, img32f);

        double minv, maxv;
        cv::minMaxLoc(img32f.reshape(1), &minv, &maxv);
        if (maxv <= 1.0 + 1e-6 && minv >= 0.0)
            img32f *= 255.0f;

        img32f.convertTo(image_u8, CV_8UC3);
        // switch back to BGR
        cv::cvtColor(image_u8, image_u8, cv::COLOR_RGB2BGR);
    }

    return image_u8;
}
