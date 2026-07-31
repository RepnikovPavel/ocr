# dots.mocr — pure C++ + CUDA inference engine
#
# Zero PyTorch / zero vLLM: safetensors weights -> bf16 forward -> greedy tokens.
# GEMM via cuBLAS (tensor-core bf16), custom CUDA kernels for everything else
# (RMSNorm, RoPE, attention, softmax, embedding scatter, SwiGLU fusion).
#
# Designed for SM89 (RTX 4070 Ti / 4090) and SM120 (RTX 5060 Ti / 5090).

cmake_minimum_required(VERSION 3.22)
project(dots_mocr_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- CUDA architecture policy -------------------------------------------------
# CUDA 13 drops Kepler/Maxwell/Pascal. Ada = sm89, Blackwell consumer = sm120.
# We pass both so the binary is portable between the 4070 Ti and 5060 Ti boxes.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 120)
endif()
set(CMAKE_CUDA_STANDARD 17)

option(DOTS_ENABLE_CUBLAS "Use cuBLAS for GEMM (recommended)" ON)
option(DOTS_BUILD_TESTS   "Build regression/unit tests"        ON)

# ---- dependencies (all header-only, vendored) --------------------------------
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# ---- the engine --------------------------------------------------------------
add_library(dots_mocr_engine STATIC
    src/tensor.cpp
    src/safetensors_loader.cpp
    src/config.cpp
    src/tokenizer.cpp
    src/image_preproc.cpp
    src/cuda_check.cu
    src/kernels.cu
    src/attention.cu
    src/vision_tower.cu
    src/llm.cu
    src/pipeline.cpp
)
target_include_directories(dots_mocr_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_compile_features(dots_mocr_engine PUBLIC cxx_std_17)

if(DOTS_ENABLE_CUBLAS)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(dots_mocr_engine PUBLIC CUDA::cublas CUDA::cudart)
    target_compile_definitions(dots_mocr_engine PUBLIC DOTS_USE_CUBLAS=1)
else()
    find_package(CUDAToolkit REQUIRED COMPONENTS cudart)
    target_link_libraries(dots_mocr_engine PUBLIC CUDA::cudart)
endif()

target_link_libraries(dots_mocr_engine PUBLIC nlohmann_json stb)

# ---- CLI executable ----------------------------------------------------------
add_executable(dots_mocr tools/dots_mocr_cli.cpp)
target_link_libraries(dots_mocr PRIVATE dots_mocr_engine)

# ---- tests -------------------------------------------------------------------
if(DOTS_BUILD_TESTS)
    enable_testing()
    add_executable(test_tokenizer   tests/test_tokenizer.cpp)
    add_executable(test_imageproc  tests/test_imageproc.cpp)
    add_executable(test_kernels     tests/test_kernels.cu)
    add_executable(test_gemm        tests/test_gemm.cu)
    add_executable(test_loader      tests/test_loader.cpp)
    add_executable(test_regression  tests/test_regression.cpp)
    add_executable(dump_vision      tests/dump_vision.cpp)
    add_executable(test_vision_attn tests/test_vision_attn.cu)
    add_executable(test_flash_real  tests/test_flash_real.cu)
    foreach(t test_tokenizer test_imageproc test_kernels test_gemm test_loader test_regression dump_vision test_vision_attn test_flash_real)
        target_link_libraries(${t} PRIVATE dots_mocr_engine)
    endforeach()
    add_test(NAME tokenizer  COMMAND test_tokenizer)
    add_test(NAME imageproc COMMAND test_imageproc)
    add_test(NAME kernels   COMMAND test_kernels)
    add_test(NAME gemm      COMMAND test_gemm)
endif()
