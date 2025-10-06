#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <nnapi_provider_factory.h>
#include <android/log.h>
#include <onnxruntime_session_options_config_keys.h>
#include "config.h"

class ModelSession;

class InferenceRunner {
public:
    InferenceRunner();

    // Provide the model path and configure internal settings
    std::vector<std::shared_ptr<ModelSession>> init_models(std::vector<std::string> model_paths, RunnerSettings s);
    std::shared_ptr<ModelSession> init_model(std::string model_path, RunnerSettings s);

private:
    void start_environment_();

    std::vector<std::string> model_paths_;
    Ort::MemoryInfo mem_info_{nullptr};
    Ort::Env env_;
};
