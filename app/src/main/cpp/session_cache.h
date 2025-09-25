

#ifndef CPPONNXRUNNER_SESSION_CACHE_H
#define CPPONNXRUNNER_SESSION_CACHE_H
#include "onnxruntime_cxx_api.h"

struct ArtifactPaths {
    std::string cache_dir_path;
    std::string inference_model_path;

    ArtifactPaths(const std::string& cache_dir_path) :
            cache_dir_path(cache_dir_path), inference_model_path(cache_dir_path + "/inference.onnx") {}
};

struct SessionCache {
    ArtifactPaths artifact_paths;
    Ort::Env ort_env;
    Ort::SessionOptions session_options;
    Ort::Session* inference_session;

    SessionCache(const std::string& cache_dir_path) :
            artifact_paths(cache_dir_path),
            ort_env(ORT_LOGGING_LEVEL_WARNING, "cpponnxrunner"), session_options(),
            inference_session(nullptr) {}
};

#endif //CPPONNXRUNNER_SESSION_CACHE_H