#include "ModelSession.h"
#include "logging.h"

/*
 * https://github.com/devingarg/onnx-quantization/blob/main/resnet_inference.cpp
 */

ModelSession::ModelSession(Ort::Env &env,
                           Ort::MemoryInfo &mem_info,
                           RunnerSettings s,
                           std::string model_path)
        : mem_info_(mem_info) {
    model_path_ = model_path;
    settings_ = s;

    Ort::SessionOptions so = init_session(s);
    session_ = Ort::Session(env, model_path_.c_str(), so);

    find_input_output_info_();
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

        __android_log_print(ANDROID_LOG_INFO, "cpponnxrunner",
                            "run(): img[%dx%d ch=%d type=%d] mask[%dx%d ch=%d type=%d] target=%dx%d | in=%zu out=%zu",
                            image.cols, image.rows, image.channels(), image.type(),
                            mask.cols, mask.rows, mask.channels(), mask.type(),
                            image_width_, image_height_,
                            input_names_.size(), output_names_.size());
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
        std::vector<Ort::Value> outputs;
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

        return output_mats;
    } catch (const Ort::Exception &e) {
        __android_log_print(ANDROID_LOG_ERROR, "cpponnxrunner",
                            "runEndToEnd Ort::Exception: %s", e.what());
        throw;
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

void ModelSession::find_input_output_info_() {
    in_count = session_.GetInputCount();
    out_count = session_.GetOutputCount();

    input_shapes_.clear();
    output_shapes_.clear();
    input_shapes_.reserve(in_count);
    output_shapes_.reserve(out_count);

    const int64_t batchSize = 1;

    for (size_t i = 0; i < in_count; ++i) {
        std::vector<int64_t> shp = getDataShape(session_.GetInputTypeInfo(i));
        if (!shp.empty() && shp[0] == -1) {
            shp[0] = batchSize;
        }
        input_shapes_.push_back(std::move(shp));
    }

    for (size_t i = 0; i < out_count; ++i) {
        std::vector<int64_t> shp = getDataShape(session_.GetOutputTypeInfo(i));
        output_shapes_.push_back(std::move(shp));
    }

    input_names_ = session_.GetInputNames();
    output_names_ = session_.GetOutputNames();
    image_width_ = static_cast<int>(input_shapes_[0][3]);
    image_height_ = static_cast<int>(input_shapes_[0][2]);

    auto shape_to_str = [](const std::vector<int64_t> &shp) {
        std::string s = "";
        s.reserve(64);
        s.push_back('[');
        for (size_t i = 0; i < shp.size(); ++i) {
            if (i) s.push_back(',');
            if (shp[i] < 0) {                // dinamik boyutlar -1/-2 vs.
                s.push_back('?');
            } else {
                s += std::to_string(shp[i]);
            }
        }
        s.push_back(']');
        return s;
    };

    LOGI("[MODEL] path='%s'", model_path_.c_str());

    LOGI("[IO] input_count=%zu, output_count=%zu", in_count, out_count);
    LOGI("[IO] target_image_size (assumed NCHW) -> W=%d H=%d", image_width_, image_height_);

// ---- LOG: tüm inputlar ----
    for (size_t i = 0; i < in_count; ++i) {
        // isim
        const char *name_c = (i < input_names_.size()) ? input_names_[i].c_str() : "<noname>";
        // eleman tipi
        Ort::TypeInfo tinfo = session_.GetInputTypeInfo(i);
        auto tinf = tinfo.GetTensorTypeAndShapeInfo();
        int etype = static_cast<int>(tinf.GetElementType());

        // shape string
        const std::string shp_str = shape_to_str(input_shapes_[i]);

        LOGI("[IN  %zu] name='%s'  elem_type=%d  shape=%s",
             i, name_c, etype, shp_str.c_str());
    }

    // ---- LOG: tüm outputlar ----
    for (size_t i = 0; i < out_count; ++i) {
        // isim
        const char *name_c = (i < output_names_.size()) ? output_names_[i].c_str() : "<noname>";
        // eleman tipi
        Ort::TypeInfo tinfo = session_.GetOutputTypeInfo(i);
        auto tinf = tinfo.GetTensorTypeAndShapeInfo();
        int etype = static_cast<int>(tinf.GetElementType());

        // shape string
        const std::string shp_str = shape_to_str(output_shapes_[i]);

        LOGI("[OUT %zu] name='%s'  elem_type=%d  shape=%s",
             i, name_c, etype, shp_str.c_str());
    }

}

std::vector<int64_t> ModelSession::getDataShape(Ort::TypeInfo info) {

    auto tensorInfo = info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensorInfo.GetShape();
    return shape;
}

cv::Mat ModelSession::ort_output_to_mat(const Ort::Value &out) {
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


Ort::SessionOptions ModelSession::init_session(RunnerSettings s) {
    Ort::SessionOptions so;

    // Threading
    so.SetInterOpNumThreads(s.num_cpu_cores);
    if (!s.use_xnnpack || s.xnnpack.use_session_threads) {
        so.SetIntraOpNumThreads(s.num_cpu_cores);
    }

    // Graph opt
    if (s.use_nnapi) {
        so.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    } else if (s.use_layout_optimization_instead_of_extended) {
        so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    } else {
        so.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    }

    if (s.use_parallel_execution) {
        so.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    }

    // XNNPACK
    if (s.use_xnnpack) {
        so.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning,
                          "0");
        if (!s.xnnpack.use_session_threads) {
            so.AppendExecutionProvider("XNNPACK",
                                       {{"intra_op_num_threads",
                                         std::to_string(s.num_cpu_cores).c_str()}});
            so.SetIntraOpNumThreads(1); // TODO 0 is faster
        } else {
            so.AppendExecutionProvider("XNNPACK", {{"intra_op_num_threads", "0"}});
        }
    }

    // NNAPI (Android)
    if (s.use_nnapi) {
        const uint32_t nnapi_flags = NnapiOptions::to_raw(s.nnapi.flags);
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(so, nnapi_flags));
    }


    return so;
}
