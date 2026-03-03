cmake_minimum_required(VERSION 3.10)
project(FastQIndexing LANGUAGES CXX)

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA using the legacy FindCUDA module
find_package(CUDA REQUIRED)


# Include CUDA and project directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(src/utils include portable psz codec/hf/include)


###############################################################################
# 1) Build the "huffman-test" executable (moved from codec/hf/src/test.cu)
###############################################################################
set(PSZ_BACKEND "CUDA" CACHE STRING "Select the desired backend: CUDA/cuda, HIP/hip and ONEAPI/1api.")

cmake_minimum_required(VERSION 3.18)
project(CUSZ LANGUAGES CUDA CXX C VERSION 0.10.0)

# general cmake setup
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "prefer shared libaries" ON)

message("[psz::info] CUDA backend has been selected.")
add_compile_definitions(PSZ_USE_CUDA)
add_compile_definitions(_PORTABLE_USE_CUDA)

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

include(GNUInstallDirs)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/psz/src/cusz_version.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/psz/include/cusz_version.h)

add_library(psz_cu_compile_settings INTERFACE)
add_library(CUSZ::compile_settings ALIAS psz_cu_compile_settings)

target_compile_definitions(
        psz_cu_compile_settings
        INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:__STRICT_ANSI__>)
target_compile_options(
        psz_cu_compile_settings
        INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda
        --expt-relaxed-constexpr -Wno-deprecated-declarations>)
target_compile_features(psz_cu_compile_settings INTERFACE cxx_std_17 cuda_std_17)

target_include_directories(
        psz_cu_compile_settings
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/portable/include/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psz/src/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psz/include/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/codec/hf/include/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/codec/hf/src/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/codec/fzg/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/codec/fzg/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz>
)

add_library(psz_cu_stat
        psz/src/stat/compare.stl.cc
        psz/src/stat/identical/all.cu
        psz/src/stat/extrema/f4.cu
        psz/src/stat/extrema/f8.cu
        psz/src/stat/calcerr/f4.cu
        psz/src/stat/calcerr/f8.cu
        psz/src/stat/assess/f4.cu
        psz/src/stat/assess/f8.cu
        psz/src/stat/maxerr/max_err.cu
)
target_link_libraries(psz_cu_stat
        PUBLIC
        psz_cu_compile_settings
)
add_library(CUSZ::stat ALIAS psz_cu_stat)

add_library(psz_cu_mem
        portable/src/mem/memobj.f.cc
        portable/src/mem/memobj.i.cc
        portable/src/mem/memobj.u.cc
        portable/src/mem/memobj.misc.cc)
target_link_libraries(psz_cu_mem
        PUBLIC
        psz_cu_compile_settings
        psz_cu_stat
        CUDA::cudart
)
add_library(CUSZ::mem ALIAS psz_cu_mem)

add_library(psz_cu_utils
        psz/src/utils/viewer.cc
        psz/src/utils/viewer.cu
        psz/src/utils/verinfo.cc
        psz/src/utils/verinfo.cu
        psz/src/utils/verinfo_nv.cu
        psz/src/utils/vis_stat.cc
        psz/src/utils/context.cc
        psz/src/utils/header.c
)
target_link_libraries(psz_cu_utils
        PUBLIC
        psz_cu_compile_settings
        CUDA::cudart CUDA::nvml
)
add_library(CUSZ::utils ALIAS psz_cu_utils)

add_library(psz_cu_phf
        codec/hf/src/hfclass.cc
        codec/hf/src/hf_est.cc
        codec/hf/src/hfbk_impl1.seq.cc
        codec/hf/src/hfbk_impl2.seq.cc
        codec/hf/src/hfbk_internal.seq.cc
        codec/hf/src/hfbk.seq.cc
        codec/hf/src/hfcanon.seq.cc
        codec/hf/src/hfcxx_module.cu
        codec/hf/src/libphf.cc
)
target_link_libraries(psz_cu_phf
        PUBLIC
        psz_cu_compile_settings
        psz_cu_stat
        CUDA::cuda_driver
)
add_library(CUSZ::phf ALIAS psz_cu_phf)

# Define the huffman-test executable
add_executable(huffman-test src/test.cu)

# Link against necessary libraries
target_link_libraries(huffman-test PRIVATE
        psz_cu_compile_settings
        CUSZ::phf
        CUSZ::stat
        CUSZ::mem
        CUSZ::utils
        CUDA::cudart
)

# Set C++ and CUDA standards
target_compile_features(huffman-test PRIVATE cxx_std_17 cuda_std_17)

# Enable CUDA separable compilation if required
set_target_properties(huffman-test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Apply CUDA-specific compile options
target_compile_options(huffman-test PRIVATE
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda
        --expt-relaxed-constexpr -Wno-deprecated-declarations>
)

# Explicitly add include directories to ensure headers are found
target_include_directories(huffman-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/codec/hf/include
        ${CMAKE_CURRENT_SOURCE_DIR}/psz/include
)

# Install the huffman-test executable
install(TARGETS huffman-test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install headers for Huffman
install(DIRECTORY
        portable/include/
        psz/include/
        codec/hf/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz
)
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/psz/include/cusz_version.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz/)



###############################################################################
# 2) Build the "main_index" executable from main_indexing.cpp
###############################################################################
cuda_add_executable(main_index
        src/main_indexing_cuda.cu
        src/indexing.cu
        src/utils/file_utils.cpp
)
target_link_libraries(main_index ${CUDA_LIBRARIES})

###############################################################################
# 3) Build the original "main" executable (CUDA + C++ source files)
###############################################################################
cuda_add_executable(main
        src/main.cu
        src/indexing.cu
        src/utils/file_utils.cpp
        src/rle.cu
        src/lorenzo.cu
        src/identifier_processing.cu
        src/identifier_parser.cu
        src/identifier_compressor.cu
        src/gpulz.cu
        src/basecall_compress.cu
        src/quality_scores_process.cu
        src/huffman_decode.cu
)

target_link_libraries(main
        ${CUDA_LIBRARIES}
        psz_cu_compile_settings
        CUSZ::phf
        CUSZ::stat
        CUSZ::mem
        CUSZ::utils
        CUDA::cudart
)

###############################################################################
# 4) Build the "main_basecall" executable (CUDA + C++ source files)
###############################################################################
cuda_add_executable(main_basecall
        src/main_basecall.cu
        src/indexing.cu
        src/utils/file_utils.cpp
        src/basecall_compress.cu
)
target_link_libraries(main_basecall ${CUDA_LIBRARIES})
