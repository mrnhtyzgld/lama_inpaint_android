#pragma once
#include <string>
#include <jni.h>
#include <vector>
#include <string>

std::vector<uint8_t> JByteArrayToVector(JNIEnv* env, jbyteArray arr);
jbyteArray VectorToJByteArray(JNIEnv* env, const std::vector<uint8_t>& v);
std::vector<std::string> JStringArrayToVector(JNIEnv* env, jobjectArray arr);
std::string JString2String(JNIEnv *env, jstring jStr);