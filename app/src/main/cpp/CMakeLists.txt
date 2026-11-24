cmake_minimum_required(VERSION 3.10)
project(dgtruss_yolo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NCNN_ROOT "${CMAKE_SOURCE_DIR}/ncnn-android-vulkan")
if(EXISTS "${NCNN_ROOT}/${ANDROID_ABI}/lib/cmake/ncnn/ncnnConfig.cmake")
    set(ncnn_DIR "${NCNN_ROOT}/${ANDROID_ABI}/lib/cmake/ncnn")
elseif(EXISTS "${NCNN_ROOT}/ncnn-20261020-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn/ncnnConfig.cmake")
    set(ncnn_DIR "${NCNN_ROOT}/ncnn-20261020-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn")
else()
    file(GLOB_RECURSE NCNN_CMAKE_FILES "${NCNN_ROOT}/*/${ANDROID_ABI}/lib/cmake/ncnn/ncnnConfig.cmake")
    if(NCNN_CMAKE_FILES)
        list(GET NCNN_CMAKE_FILES 0 NCNN_FIRST_CMAKE)
        get_filename_component(ncnn_DIR "${NCNN_FIRST_CMAKE}" DIRECTORY)
    else()
        message(FATAL_ERROR "ncnnConfig.cmake not found for ABI ${ANDROID_ABI} under ${NCNN_ROOT}")
    endif()
endif()

find_package(ncnn REQUIRED)

add_library(dgtruss_yolo SHARED yolo_jni.cpp)

target_compile_options(dgtruss_yolo PRIVATE
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    -ffunction-sections
    -fdata-sections
)

target_link_options(dgtruss_yolo PRIVATE
    "-Wl,-z,max-page-size=16384"
    "-Wl,--gc-sections"
    "-Wl,--exclude-libs,ALL"
)

target_link_libraries(dgtruss_yolo
    ncnn
    android
    jnigraphics
)
