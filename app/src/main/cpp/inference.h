#ifndef CPPONNXRUNNER_INFERENCE_H
#define CPPONNXRUNNER_INFERENCE_H

#include <cstdint>
#include <vector>
#include <string>
#include "session_cache.h"

namespace inference {

// image: 1x3xHxW, mask: 1x1xHxW (float32), NCHW
    std::vector<float> infer(
            SessionCache* session_cache,
            float* image_data,
            float* mask_data,
            int64_t batch_size,
            int64_t image_channels,    // 3
            int64_t image_rows,
            int64_t image_cols
    );

} // namespace inference

#endif // CPPONNXRUNNER_INFERENCE_H
