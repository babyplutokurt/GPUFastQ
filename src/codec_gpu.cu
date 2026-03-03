#include "gpufastq/codec_gpu.cuh"

#include <cub/device/device_adjacent_difference.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace gpufastq {

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +      \
                               ":" + std::to_string(__LINE__) + ": " +         \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

namespace {

template <typename T> void cuda_free_if_set(T *ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

__global__ void cast_u64_to_u32_kernel(const uint64_t *input, uint32_t *output,
                                       size_t count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  const uint64_t value = input[idx];
  if (value > 0xFFFFFFFFull) {
    asm("trap;");
  }
  output[idx] = static_cast<uint32_t>(value);
}

__global__ void widen_u32_to_u64_kernel(const uint32_t *input, uint64_t *output,
                                        size_t count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  output[idx] = static_cast<uint64_t>(input[idx]);
}

} // namespace

void delta_encode_offsets_to_lengths(const uint64_t *d_offsets,
                                     uint32_t *d_lengths, size_t count,
                                     cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  if (d_offsets == nullptr || d_lengths == nullptr) {
    throw std::runtime_error("Delta-encode pointers must not be null");
  }

  uint64_t *d_deltas64 = nullptr;
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  try {
    CUDA_CHECK(cudaMalloc(&d_deltas64, count * sizeof(uint64_t)));

    CUDA_CHECK(cub::DeviceAdjacentDifference::SubtractLeftCopy(
        nullptr, temp_storage_bytes, d_offsets, d_deltas64, count,
        cub::Difference{}, stream));
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CUDA_CHECK(cub::DeviceAdjacentDifference::SubtractLeftCopy(
        d_temp_storage, temp_storage_bytes, d_offsets, d_deltas64, count,
        cub::Difference{}, stream));

    const uint32_t block_size = 256;
    const uint32_t grid_size =
        static_cast<uint32_t>((count + block_size - 1) / block_size);
    cast_u64_to_u32_kernel<<<grid_size, block_size, 0, stream>>>(
        d_deltas64, d_lengths, count);
    CUDA_CHECK(cudaGetLastError());
  } catch (...) {
    cuda_free_if_set(static_cast<char *>(d_temp_storage));
    cuda_free_if_set(d_deltas64);
    throw;
  }

  cuda_free_if_set(static_cast<char *>(d_temp_storage));
  cuda_free_if_set(d_deltas64);
}

void delta_decode_lengths_to_offsets(const uint32_t *d_lengths,
                                     uint64_t *d_offsets, size_t count,
                                     cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  if (d_lengths == nullptr || d_offsets == nullptr) {
    throw std::runtime_error("Delta-decode pointers must not be null");
  }

  uint64_t *d_lengths64 = nullptr;
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  try {
    CUDA_CHECK(cudaMalloc(&d_lengths64, count * sizeof(uint64_t)));

    const uint32_t block_size = 256;
    const uint32_t grid_size =
        static_cast<uint32_t>((count + block_size - 1) / block_size);
    widen_u32_to_u64_kernel<<<grid_size, block_size, 0, stream>>>(
        d_lengths, d_lengths64, count);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes,
                                             d_lengths64, d_offsets, count,
                                             stream));
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                             d_lengths64, d_offsets, count,
                                             stream));
  } catch (...) {
    cuda_free_if_set(static_cast<char *>(d_temp_storage));
    cuda_free_if_set(d_lengths64);
    throw;
  }

  cuda_free_if_set(static_cast<char *>(d_temp_storage));
  cuda_free_if_set(d_lengths64);
}

} // namespace gpufastq
