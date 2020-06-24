# Sample CMakeLists.txt file for an OpenVINO Inference Engine project
cmake_minimum_required (VERSION 2.8.1)

project(onnx-importer)
set(TARGET_NAME onnx-importer)     # name of executable file
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIE")

find_package(InferenceEngine 1.1 REQUIRED)
find_package(OpenCV REQUIRED)
add_definitions(-DUSE_OPENCV)

set(nGRAPH_INC_DIRS "$ENV{INTEL_OPENVINO_DIR}/deployment_tools/ngraph/include")
set(nGRAPH_LIB_DIRS "$ENV{INTEL_OPENVINO_DIR}/deployment_tools/ngraph/lib")
if(UNIX)
  set(nGRAPH_LIBS "ngraph")
elseif(WIN32)
  set(nGRAPH_LIBS "ngraph.lib")
endif(UNIX)

include_directories( ${InferenceEngine_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${nGRAPH_INC_DIRS})
link_directories( ${nGRAPH_LIB_DIRS} )

add_executable( ${TARGET_NAME} main.cpp )    # list of source file(s)
set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")
target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${OpenCV_LIBS} ${nGRAPH_LIBS})
