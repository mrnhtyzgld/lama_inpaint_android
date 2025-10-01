#include <jni.h>
#include <string>
#include <vector>

#include "utils.h"
#include "onnxruntime_cxx_api.h"
#include "InferenceRunner.h"
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <jni.h>
#include <string>

InferenceRunner runner;

inline std::string JStringToString(JNIEnv *env, jstring jstr) {
    if (!jstr) return {};
    const char *utf = env->GetStringUTFChars(jstr, nullptr); // JNI modified-UTF8
    std::string out = utf ? std::string(utf) : std::string();
    if (utf) env->ReleaseStringUTFChars(jstr, utf);
    return out;
}

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
        return nullptr; // or you can throw a Java exception here
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

extern "C" JNIEXPORT void JNICALL
Java_com_example_cpponnxrunner_MainActivity_createSession(
        JNIEnv *env, jobject /* this */,
        jstring cache_dir_path) {

    runner.init_model(JStringToString(env, cache_dir_path));
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_cpponnxrunner_MainActivity_releaseSession(
        JNIEnv * /*env*/, jobject /* this */) {
    runner.end_profiling_and_log();
    return;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_cpponnxrunner_MainActivity_cvVersion(JNIEnv *env, jobject) {
    std::string ver = cv::getVersionString();
    return env->NewStringUTF(ver.c_str());
}

extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_example_cpponnxrunner_MainActivity_inferFromBytes(JNIEnv *env, jobject thiz,
                                                           jbyteArray image_bytes,
                                                           jbyteArray mask_bytes) {
    return VectorToJByteArray(env, runner.runByteToByte(JByteArrayToVector(env, image_bytes),
                                                        JByteArrayToVector(env, mask_bytes)));

}