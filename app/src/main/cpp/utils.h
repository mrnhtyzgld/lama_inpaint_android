

#ifndef CPPONNXRUNNER_UTILS_H
#define CPPONNXRUNNER_UTILS_H

#include <string>
#include <jni.h>

namespace utils {

    // Convert jstring to std::string
    std::string JString2String(JNIEnv *env, jstring jStr);

} // utils

#endif //CPPONNXRUNNER_UTILS_H
