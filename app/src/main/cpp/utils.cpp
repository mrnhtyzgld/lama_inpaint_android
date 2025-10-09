#include "utils.h"
#include <limits>

std::vector<uint8_t> JByteArrayToVector(JNIEnv *env, jbyteArray arr) {
    if (!arr) return {};
    const jsize len = env->GetArrayLength(arr);
    std::vector<uint8_t> out(static_cast<size_t>(len));
    if (len > 0) {
        env->GetByteArrayRegion(arr, 0, len,
                                reinterpret_cast<jbyte *>(out.data()));
    }
    return out;
}

jbyteArray VectorToJByteArray(JNIEnv *env, const std::vector<uint8_t> &v) {
    if (!env) return nullptr;
    if (v.size() > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
        return nullptr;
    }
    jsize len = static_cast<jsize>(v.size());
    jbyteArray arr = env->NewByteArray(len);
    if (!arr) return nullptr;
    if (len > 0) {
        env->SetByteArrayRegion(arr, 0, len,
                                reinterpret_cast<const jbyte *>(v.data()));
    }
    return arr;
}

std::vector<std::string> JStringArrayToVector(JNIEnv *env, jobjectArray arr) {
    std::vector<std::string> out;
    if (!arr) return out;
    jsize n = env->GetArrayLength(arr);
    out.reserve(n);
    for (jsize i = 0; i < n; ++i) {
        jstring jstr = static_cast<jstring>(env->GetObjectArrayElement(arr, i));
        if (!jstr) {
            out.emplace_back("");
            continue;
        }
        const char *cstr = env->GetStringUTFChars(jstr, nullptr);
        out.emplace_back(cstr ? cstr : "");
        if (cstr) env->ReleaseStringUTFChars(jstr, cstr);
        env->DeleteLocalRef(jstr);
    }
    return out;
}


std::string JString2String(JNIEnv *env, jstring jStr) {
    if (!jStr)
        return {};

    const jclass stringClass = env->GetObjectClass(jStr);
    const jmethodID getBytes = env->GetMethodID(stringClass, "getBytes",
                                                "(Ljava/lang/String;)[B");
    const jbyteArray stringJbytes = (jbyteArray) env->CallObjectMethod(jStr, getBytes,
                                                                       env->NewStringUTF(
                                                                               "UTF-8"));

    size_t length = (size_t) env->GetArrayLength(stringJbytes);
    jbyte *pBytes = env->GetByteArrayElements(stringJbytes, nullptr);

    std::string ret = std::string((char *) pBytes, length);
    env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

    env->DeleteLocalRef(stringJbytes);
    env->DeleteLocalRef(stringClass);
    return ret;
}

