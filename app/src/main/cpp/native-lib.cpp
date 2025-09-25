#include <jni.h>
#include <string>
#include <vector>

#include "session_cache.h"
#include "utils.h"
#include "inference.h"
#include "onnxruntime_cxx_api.h"

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_cpponnxrunner_MainActivity_createSession(
        JNIEnv *env, jobject /* this */,
        jstring cache_dir_path)
{
    std::unique_ptr<SessionCache> session_cache =
            std::make_unique<SessionCache>(utils::JString2String(env, cache_dir_path));
    return reinterpret_cast<jlong>(session_cache.release());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_cpponnxrunner_MainActivity_releaseSession(
        JNIEnv * /*env*/, jobject /* this */,
        jlong session)
{
    auto *session_cache = reinterpret_cast<SessionCache *>(session);
    if (!session_cache) return;
    delete session_cache->inference_session; // nullptr ise güvenli
    session_cache->inference_session = nullptr;
    delete session_cache;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_cpponnxrunner_MainActivity_performInference(
        JNIEnv* env, jobject /* this */,
        jlong session,
        jfloatArray image_buffer,   // 1x3xHxW
        jfloatArray mask_buffer,    // 1x1xHxW
        jint batch_size, jint channels, jint width, jint height)
{
    auto* session_cache = reinterpret_cast<SessionCache*>(session);

    // inference_session yoksa oluştur
    if (!session_cache->inference_session) {
        const char* model_path = session_cache->artifact_paths.inference_model_path.c_str();
        session_cache->inference_session = new Ort::Session(
                session_cache->ort_env, model_path, session_cache->session_options
        );
    }

    // Java FloatArray -> native pointer'lar
    jboolean isCopyImg = JNI_FALSE, isCopyMsk = JNI_FALSE;
    jfloat* img_ptr  = env->GetFloatArrayElements(image_buffer, &isCopyImg);
    jfloat* msk_ptr  = env->GetFloatArrayElements(mask_buffer,  &isCopyMsk);

    // Inference: std::vector<float> döner
    // Not: infer imzan; (session_cache, image_data, mask_data, batch, image_channels, rows, cols)
    std::vector<float> out = inference::infer(
            session_cache,
            reinterpret_cast<float*>(img_ptr),
            reinterpret_cast<float*>(msk_ptr),
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(channels),     // image channels (=3)
            static_cast<int64_t>(height),       // rows (H)
            static_cast<int64_t>(width)         // cols (W)
    );

    // Input'ları bırak
    env->ReleaseFloatArrayElements(image_buffer, img_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(mask_buffer,  msk_ptr, JNI_ABORT);

    // std::vector<float> -> jfloatArray
    jfloatArray j_out = env->NewFloatArray(static_cast<jsize>(out.size()));
    if (!j_out) return nullptr;
    env->SetFloatArrayRegion(j_out, 0, static_cast<jsize>(out.size()), out.data());
    return j_out;
}
