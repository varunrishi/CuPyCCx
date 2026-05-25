cmake_minimum_required(VERSION 3.20)

# Force BSD-format archiver on macOS to avoid GNU ar format incompatibility with ld
if(APPLE)
    find_program(_SYS_AR ar PATHS /usr/bin NO_DEFAULT_PATH)
    find_program(_SYS_RANLIB ranlib PATHS /usr/bin NO_DEFAULT_PATH)
    if(_SYS_AR)
        set(CMAKE_AR "${_SYS_AR}" CACHE FILEPATH "Archiver" FORCE)
    endif()
    if(_SYS_RANLIB)
        set(CMAKE_RANLIB "${_SYS_RANLIB}" CACHE FILEPATH "Ranlib" FORCE)
    endif()
endif()

project(cupyccx
    VERSION 0.1
    DESCRIPTION "Coupled Cluster Doubles: C++/CUDA/Python"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(CUPYCCX_BUILD_PYTHON "Build Python bindings (requires pybind11)" ON)
option(CUPYCCX_BUILD_TESTS  "Build C++ unit tests"                       ON)
option(CUPYCCX_CUDA         "Enable CUDA GPU backend"                    OFF)
option(CUPYCCX_USE_OPENMP   "Enable OpenMP threading"                    ON)


# ---------------------------------------------------------------------------
# C++ standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

# Emit compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Default build type
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# CUDA (optional)
# ---------------------------------------------------------------------------
if(CUPYCCX_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD          17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(CUPYCCX_CUDA)
    # Default to sm_70 (Volta); override with -DCUPYCCX_CUDA_ARCH=...
    if(NOT DEFINED CUPYCCX_CUDA_ARCH)
        set(CUPYCCX_CUDA_ARCH "70")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${CUPYCCX_CUDA_ARCH})
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA enabled — architecture sm_${CUPYCCX_CUDA_ARCH}")
endif()

# ---------------------------------------------------------------------------
# External dependencies
# ---------------------------------------------------------------------------

# Eigen (header-only linear algebra)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")

# OpenMP (optional parallelism)
if(CUPYCCX_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP enabled")
    else()
        message(WARNING "OpenMP not found — building without threading")
    endif()
endif()

# pybind11 (fetched from PyPI via scikit-build or found locally)
if(CUPYCCX_BUILD_PYTHON)
    find_package(pybind11 CONFIG)
    if(NOT pybind11_FOUND)
        # Try fetching via FetchContent
        include(FetchContent)
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.12.0)
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# ---------------------------------------------------------------------------
# Core library (header-only interface + compiled implementation)
# ---------------------------------------------------------------------------
add_library(cupyccx_core STATIC
    src/cpp/ccd_base.cpp
    src/cpp/ccd.cpp
    src/cpp/tensor_ops_cpu.cpp
)

target_include_directories(cupyccx_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cupyccx_core PUBLIC
    Eigen3::Eigen
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cupyccx_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# CUDA library (compiled only when CUPYCCX_CUDA=ON)
# ---------------------------------------------------------------------------
if(CUPYCCX_CUDA)
    add_library(cupyccx_cuda STATIC
        src/cuda/tensor_ops_gpu.cu
    )
    target_include_directories(cupyccx_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
    )
    target_link_libraries(cupyccx_cuda PUBLIC
        CUDA::cublas
        Eigen3::Eigen
    )
    # Set CUDA-specific compile options
    set_target_properties(cupyccx_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE  ON
    )
    target_link_libraries(cupyccx_core PUBLIC cupyccx_cuda)
endif()

# Make the core library position-independent (needed for shared libraries)
set_target_properties(cupyccx_core PROPERTIES POSITION_INDEPENDENT_CODE ON)


# ---------------------------------------------------------------------------
# Python extension module
# ---------------------------------------------------------------------------
if(CUPYCCX_BUILD_PYTHON)
    pybind11_add_module(_cupyccx MODULE
        src/python/bindings.cpp
    )
    target_link_libraries(_cupyccx PRIVATE cupyccx_core)
    target_include_directories(_cupyccx PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    install(TARGETS _cupyccx DESTINATION cupyccx)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(CUPYCCX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------
install(DIRECTORY include/ DESTINATION include)
install(TARGETS cupyccx_core ARCHIVE DESTINATION lib)
