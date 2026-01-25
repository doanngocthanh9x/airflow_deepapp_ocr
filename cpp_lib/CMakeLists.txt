# CMake config for cpp_lib: C++17, ONNX, pybind11, đồng bộ version Python
cmake_minimum_required(VERSION 3.15)
project(DeepAppOCR_CPP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find OpenCV (optional, nếu dùng)
# find_package(OpenCV 4 REQUIRED)

# Find Python3 (for pybind11)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 CONFIG QUIET)

# ONNX Runtime (manual path, chỉnh lại nếu cần)
set(ONNXRUNTIME_ROOT_PATH "/usr/local" CACHE PATH "Path to ONNXRuntime")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so")

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${ONNXRUNTIME_INCLUDE_DIRS}
	${Python3_INCLUDE_DIRS}
)

# Core C++ library (ví dụ)
# add_library(deepapp_ocr_cpp SHARED src/your_code.cpp)
# target_link_libraries(deepapp_ocr_cpp ${ONNXRUNTIME_LIB})

# Python bindings with pybind11
if(pybind11_FOUND)
	pybind11_add_module(deepapp_cpp_bindings python/bindings.cpp)
	target_link_libraries(deepapp_cpp_bindings PRIVATE ${ONNXRUNTIME_LIB})
	set_target_properties(deepapp_cpp_bindings PROPERTIES
		OUTPUT_NAME "deepapp_cpp"
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python"
	)
else()
	message(WARNING "pybind11 not found. Python bindings will not be built.")
	message(STATUS "Install pybind11 with: pip install pybind11")
endif()

# Gợi ý: Muốn đồng bộ version, hãy kiểm tra version onnx, torch, numpy trong Python và cài đúng bản onnxruntime C++ tương ứng.
