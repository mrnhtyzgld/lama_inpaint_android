#include <jni.h>
#include <string>
#include <vector>
#include <thread>
#include <limits>

#include "utils.h"
#include "onnxruntime_cxx_api.h"
#include "InferenceRunner.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>
#include <jni.h>
#include <string>

#include "onnxruntime_cxx_api.h"

#include "InferenceRunner.h"
#include "ModelSession.h"

// -------------------- Global --------------------
static InferenceRunner g_runner; // tek Env + MemInfo
static std::shared_ptr<ModelSession> g_modelA;
static std::shared_ptr<ModelSession> g_modelB;

// -------------------- JNI helpers --------------------
static std::vector<uint8_t> JByteArrayToVector(JNIEnv *env, jbyteArray arr) {
    if (!arr) return {};
    const jsize len = env->GetArrayLength(arr);
    std::vector<uint8_t> out(static_cast<size_t>(len));
    if (len > 0) {
        // jbyte (signed) -> uint8_t (unsigned) but bit-level is identical; reinterpret is safe
        env->GetByteArrayRegion(arr, 0, len, reinterpret_cast<jbyte *>(out.data()));
    }
    return out;
}

static jbyteArray VectorToJByteArray(JNIEnv* env, const std::vector<uint8_t>& v) {
    if (!env) return nullptr;

    // check jsize limit (important on 32-bit JVMs)
    if (v.size() > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
        return nullptr;
    }

    jsize len = static_cast<jsize>(v.size());
    jbyteArray arr = env->NewByteArray(len);
    if (!arr) return nullptr;

    if (len > 0) {
        env->SetByteArrayRegion(arr, 0, len,
                                reinterpret_cast<const jbyte*>(v.data()));
    }
    return arr;
}

static std::vector<std::string> JStringArrayToVector(JNIEnv* env, jobjectArray arr) {
    std::vector<std::string> out;
    jsize n = env->GetArrayLength(arr);
    out.reserve(n);
    for (jsize i = 0; i < n; ++i) {
        jstring jstr = (jstring) env->GetObjectArrayElement(arr, i);
        const char* cstr = env->GetStringUTFChars(jstr, nullptr);
        out.emplace_back(cstr ? cstr : "");
        env->ReleaseStringUTFChars(jstr, cstr);
        env->DeleteLocalRef(jstr);
    }
    return out;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_cpponnxrunner_MainActivity_createSession(JNIEnv* env, jobject thiz, jobjectArray modelPaths) {
    auto paths = JStringArrayToVector(env, modelPaths);

    RunnerSettings s{};
    s.num_cpu_cores = 8;
    s.use_xnnpack   = true;
    s.use_nnapi     = true;

    auto models = g_runner.init_models(paths, s);

    g_modelA = models[0];
    g_modelB = models[1];

}

extern "C" JNIEXPORT void JNICALL
Java_com_example_cpponnxrunner_MainActivity_releaseSession(
        JNIEnv * /*env*/, jobject /* this */) {
    return;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_cpponnxrunner_MainActivity_cvVersion(JNIEnv *env, jobject) {
    std::string ver = cv::getVersionString();
    return env->NewStringUTF(ver.c_str());
}

extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_example_cpponnxrunner_MainActivity_inferFromBytes(JNIEnv *env, jobject thiz,
                                                           jbyteArray image_bytes,
                                                           jbyteArray mask_bytes) {

    if (!g_modelA || !g_modelB) return nullptr;

    // Decode
    std::vector<uint8_t> imgV  = JByteArrayToVector(env, image_bytes);
    std::vector<uint8_t> maskV = JByteArrayToVector(env, mask_bytes);

    std::vector<uint8_t> pngBytes_1;
    //std::vector<uint8_t> pngBytes_2;
    try {
        pngBytes_1 = g_modelA->runEndToEnd(imgV, maskV);
        //pngBytes_2 = g_modelB->runEndToEnd(imgV,maskV);
    } catch (...) {
        return nullptr;
    }

    return VectorToJByteArray(env, pngBytes_1);
}
