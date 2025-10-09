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
#include <chrono>


// -------------------- Global --------------------
static InferenceRunner g_runner; // tek Env + MemInfo
static std::shared_ptr<ModelSession> g_modelA;
static std::shared_ptr<ModelSession> g_modelB;

extern "C" JNIEXPORT void JNICALL
Java_com_example_cpponnxrunner_MainActivity_createSession(JNIEnv* env, jobject thiz, jobjectArray modelPaths) {
    auto paths = JStringArrayToVector(env, modelPaths);

    RunnerSettings s{};
    s.num_cpu_cores = 4;
    s.use_xnnpack   = false;
    s.use_nnapi     = false;
    s.use_layout_optimization_instead_of_extended = true;

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
Java_com_example_cpponnxrunner_MainActivity_inferFromBytesParallel(JNIEnv *env, jobject thiz,
                                                           jbyteArray image_bytes,
                                                           jbyteArray mask_bytes) {
    if (!g_modelA || !g_modelB) return nullptr;

    // Decode
    std::vector<uint8_t> imgV  = JByteArrayToVector(env, image_bytes);
    std::vector<uint8_t> maskV = JByteArrayToVector(env, mask_bytes);

    using clock = std::chrono::steady_clock;

    std::vector<uint8_t> pngBytes_1;
    std::vector<uint8_t> pngBytes_2;

    std::exception_ptr ex1 = nullptr, ex2 = nullptr;

    long long t1_ms = -1, t2_ms = -1;  // süreler

    // ---- start gate ----
    std::mutex gate_m;
    std::condition_variable gate_cv;
    int  ready = 0;
    bool go = false;

    std::thread th1([&](){
        // hazır olduğunu bildir, start sinyalini bekle
        {
            std::unique_lock<std::mutex> lk(gate_m);
            ++ready;
            gate_cv.notify_all();
            gate_cv.wait(lk, [&]{ return go; });
        }
        auto t0 = clock::now();
        __android_log_print(ANDROID_LOG_INFO, "cpponnxrunner", "T1 start (modelA)");
        try {
            pngBytes_1 = g_modelA->runEndToEnd(imgV, maskV);
        } catch (...) {
            ex1 = std::current_exception();
        }
        auto t1 = clock::now();
        t1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        __android_log_print(ANDROID_LOG_INFO, "cpponnxrunner", "T1 end (modelA), dt=%lld ms", t1_ms);
    });

    std::thread th2([&](){
        {
            std::unique_lock<std::mutex> lk(gate_m);
            ++ready;
            gate_cv.notify_all();
            gate_cv.wait(lk, [&]{ return go; });
        }
        auto t0 = clock::now();
        __android_log_print(ANDROID_LOG_INFO, "cpponnxrunner", "T2 start (modelB)");
        try {
            pngBytes_2 = g_modelB->runEndToEnd(imgV, maskV);
        } catch (...) {
            ex2 = std::current_exception();
        }
        auto t1 = clock::now();
        t2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        __android_log_print(ANDROID_LOG_INFO, "cpponnxrunner", "T2 end (modelB), dt=%lld ms", t2_ms);
    });

    // iki thread de "hazır" diyene kadar bekle, sonra aynı anda başlat
    std::chrono::steady_clock::time_point t_all_start;
    {
        std::unique_lock<std::mutex> lk(gate_m);
        gate_cv.wait(lk, [&]{ return ready == 2; });
        t_all_start = clock::now();
        go = true;
        lk.unlock();
        gate_cv.notify_all();
    }

    th1.join();
    th2.join();

    auto t_all_end = clock::now();
    auto all_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_all_end - t_all_start).count();
    __android_log_print(ANDROID_LOG_INFO, "cpponnxrunner", "Both threads finished, total=%lld ms (t1=%lld, t2=%lld)",
                        all_ms, t1_ms, t2_ms);

    // Hata logları
    if (ex1) {
        try { std::rethrow_exception(ex1); }
        catch (const std::exception& e) {
            __android_log_print(ANDROID_LOG_ERROR, "cpponnxrunner", "T1 exception: %s", e.what());
        } catch (...) {
            __android_log_print(ANDROID_LOG_ERROR, "cpponnxrunner", "T1 exception: <unknown>");
        }
    }
    if (ex2) {
        try { std::rethrow_exception(ex2); }
        catch (const std::exception& e) {
            __android_log_print(ANDROID_LOG_ERROR, "cpponnxrunner", "T2 exception: %s", e.what());
        } catch (...) {
            __android_log_print(ANDROID_LOG_ERROR, "cpponnxrunner", "T2 exception: <unknown>");
        }
    }

    // Birinin çıktısını dön (UI tek ByteArray bekliyor)
    if (!pngBytes_1.empty()) return VectorToJByteArray(env, pngBytes_1);
    if (!pngBytes_2.empty()) return VectorToJByteArray(env, pngBytes_2);
    return nullptr;
}

extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_example_cpponnxrunner_MainActivity_inferFromBytes(JNIEnv *env, jobject thiz,
                                                           jbyteArray image_bytes,
                                                           jbyteArray mask_bytes) {


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


