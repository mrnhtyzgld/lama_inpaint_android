# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("cpponnxrunner")


set(OpenCV_DIR "C:/noWhiteSpace/packages/OpenCV-android-sdk/sdk/native/jni")
find_package(OpenCV REQUIRED)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
#
# In this top level CMakeLists.txt, ${CMAKE_PROJECT_NAME} is used to define
# the target library name; in the sub-module's CMakeLists.txt, ${PROJECT_NAME}
# is preferred for the same purpose.
#
# In order to load a library into your app from Java/Kotlin, you must call
# System.loadLibrary() and pass the name of the library defined here;
# for GameActivity/NativeActivity derived applications, the same library name must be
# used in the AndroidManifest.xml file.
add_library("cpponnxrunner" SHARED
        native-lib.cpp
        utils.cpp
        inference.cpp
        InferenceRunner.cpp
)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lib/libonnxruntime.so)
target_include_directories("cpponnxrunner" PRIVATE
        ${CMAKE_SOURCE_DIR}/include/onnxruntime
        ${OpenCV_INCLUDE_DIRS})

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(
        "cpponnxrunner"
        ${android-lib}
        ${log-lib}
        onnxruntime
        ${OpenCV_LIBS})