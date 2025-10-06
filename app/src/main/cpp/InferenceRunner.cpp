#include "InferenceRunner.h"
#include "ModelSession.h"

// TODO save optimized graph for fast load?

InferenceRunner::InferenceRunner()
        : env_(ORT_LOGGING_LEVEL_WARNING, "cpponnxrunner") {
    start_environment_();
}

std::vector<std::shared_ptr<ModelSession>>
InferenceRunner::init_models(const std::vector<std::string> model_paths,
                             const RunnerSettings s) {
    if (model_paths.empty()) throw std::invalid_argument("init_models: empty model_paths");

    model_paths_.reserve(model_paths_.size() + model_paths.size());
    model_paths_.insert(model_paths_.end(), model_paths.begin(), model_paths.end());

    std::vector<std::shared_ptr<ModelSession>> out;
    out.reserve(model_paths.size());
    for (const auto &p: model_paths) {
        out.emplace_back(std::make_shared<ModelSession>(env_, mem_info_, s, p));
    }
    return out;
}

std::shared_ptr<ModelSession>
InferenceRunner::init_model(const std::string model_path,
                            const RunnerSettings s) {
    if (model_path.empty()) throw std::invalid_argument("init_model: empty model_path");
    model_paths_.push_back(model_path);
    return std::make_shared<ModelSession>(env_, mem_info_, s, model_path);
}

void InferenceRunner::start_environment_() {
    mem_info_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
}
