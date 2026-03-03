#include "gpufastq/codec_gpu.cuh"
#include "gpufastq/codec_gpu_nvcomp.cuh"
#include "gpufastq/fastq_parser.hpp"
#include "gpufastq/gpu_compressor.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

#include <libbsc/libbsc.h>

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

constexpr size_t MAX_FIELD_SLICE_SIZE = 16 * 1024 * 1024;

struct ChunkedCompressedBuffer {
  std::vector<uint8_t> data;
  std::vector<uint64_t> chunk_sizes;
};

struct DeviceFieldBuffers {
  uint8_t *identifiers = nullptr;
  uint8_t *basecalls = nullptr;
  uint8_t *quality_scores = nullptr;
};

uint64_t sum_sizes(const std::vector<uint64_t> &sizes) {
  return std::accumulate(sizes.begin(), sizes.end(), uint64_t{0});
}

template <typename T> void cuda_free_if_set(T *ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

__global__ void compute_field_lengths_kernel(const uint64_t *line_offsets,
                                             uint64_t *identifier_lengths,
                                             uint64_t *basecall_lengths,
                                             uint64_t *quality_lengths,
                                             uint64_t num_records) {
  const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_records) {
    return;
  }

  const uint64_t id_line = 4 * idx;
  const uint64_t seq_line = id_line + 1;
  const uint64_t plus_line = id_line + 2;
  const uint64_t qual_line = id_line + 3;

  identifier_lengths[idx] = line_offsets[seq_line] - line_offsets[id_line] - 2;
  basecall_lengths[idx] = line_offsets[plus_line] - line_offsets[seq_line] - 1;
  quality_lengths[idx] =
      line_offsets[qual_line + 1] - line_offsets[qual_line] - 1;
}

__global__ void gather_fields_kernel(
    const uint8_t *raw_bytes, const uint64_t *line_offsets,
    const uint64_t *identifier_offsets, const uint64_t *basecall_offsets,
    const uint64_t *quality_offsets, const uint64_t *identifier_lengths,
    const uint64_t *basecall_lengths, const uint64_t *quality_lengths,
    uint8_t *identifiers, uint8_t *basecalls, uint8_t *quality_scores,
    uint64_t num_records) {
  const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_records) {
    return;
  }

  const uint64_t id_line = 4 * idx;
  const uint64_t seq_line = id_line + 1;
  const uint64_t qual_line = id_line + 3;

  const uint64_t id_src = line_offsets[id_line] + 1;
  const uint64_t seq_src = line_offsets[seq_line];
  const uint64_t qual_src = line_offsets[qual_line];

  const uint64_t id_dst = identifier_offsets[idx];
  const uint64_t seq_dst = basecall_offsets[idx];
  const uint64_t qual_dst = quality_offsets[idx];

  for (uint64_t i = 0; i < identifier_lengths[idx]; ++i) {
    identifiers[id_dst + i] = raw_bytes[id_src + i];
  }
  for (uint64_t i = 0; i < basecall_lengths[idx]; ++i) {
    basecalls[seq_dst + i] = raw_bytes[seq_src + i];
  }
  for (uint64_t i = 0; i < quality_lengths[idx]; ++i) {
    quality_scores[qual_dst + i] = raw_bytes[qual_src + i];
  }
}

ChunkedCompressedBuffer bsc_compress_host_chunked(const uint8_t *h_input,
                                                  size_t input_size,
                                                  size_t chunk_size) {
  ChunkedCompressedBuffer result;
  if (input_size == 0) {
    return result;
  }

  bsc_init(LIBBSC_FEATURE_FASTMODE);

  for (size_t offset = 0; offset < input_size; offset += chunk_size) {
    int target_chunk_size =
        static_cast<int>(std::min(chunk_size, input_size - offset));
    int max_compressed_size = target_chunk_size + LIBBSC_HEADER_SIZE;
    std::vector<uint8_t> compressed(max_compressed_size);

    int compressed_size =
        bsc_compress(h_input + offset, compressed.data(), target_chunk_size, 16,
                     128, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_ADAPTIVE,
                     LIBBSC_FEATURE_FASTMODE);

    if (compressed_size < LIBBSC_NO_ERROR) {
      throw std::runtime_error("BSC compression failed with code: " +
                               std::to_string(compressed_size));
    }

    result.chunk_sizes.push_back(compressed_size);
    result.data.insert(result.data.end(), compressed.data(),
                       compressed.data() + compressed_size);
  }
  return result;
}

std::vector<uint8_t>
bsc_decompress_host_chunked(const std::vector<uint8_t> &compressed,
                            const std::vector<uint64_t> &chunk_sizes,
                            uint64_t expected_size) {

  if (compressed.empty())
    return {};

  bsc_init(LIBBSC_FEATURE_FASTMODE);

  std::vector<uint8_t> output(expected_size);
  size_t compressed_offset = 0;
  size_t uncompressed_offset = 0;

  for (uint64_t chunk_size : chunk_sizes) {
    int expected_chunk_size = static_cast<int>(std::min<uint64_t>(
        8 * 1024 * 1024, expected_size - uncompressed_offset));

    int decomp_result = bsc_decompress(
        compressed.data() + compressed_offset, static_cast<int>(chunk_size),
        output.data() + uncompressed_offset, expected_chunk_size,
        LIBBSC_FEATURE_FASTMODE);

    if (decomp_result != LIBBSC_NO_ERROR && decomp_result < 0) {
      throw std::runtime_error("BSC decompression failed with code: " +
                               std::to_string(decomp_result));
    }

    compressed_offset += chunk_size;
    uncompressed_offset += expected_chunk_size;
  }
  return output;
}

ChunkedCompressedBuffer gpu_compress_device_chunked(const uint8_t *d_input,
                                                    size_t input_size,
                                                    size_t field_slice_size,
                                                    size_t nvcomp_chunk_size,
                                                    cudaStream_t stream) {
  ChunkedCompressedBuffer result;
  if (input_size == 0) {
    return result;
  }
  if (field_slice_size == 0 || field_slice_size > MAX_FIELD_SLICE_SIZE) {
    throw std::runtime_error(
        "Requested field slice size is out of supported range");
  }

  for (size_t offset = 0; offset < input_size; offset += field_slice_size) {
    const size_t slice_size = std::min(field_slice_size, input_size - offset);
    auto compressed = nvcomp_zstd_compress_device(d_input + offset, slice_size,
                                                  nvcomp_chunk_size, stream);
    result.chunk_sizes.push_back(compressed.payload.size());
    result.data.insert(result.data.end(), compressed.payload.begin(),
                       compressed.payload.end());
  }

  return result;
}

std::vector<uint8_t>
gpu_decompress_chunked(const std::vector<uint8_t> &compressed,
                       const std::vector<uint64_t> &chunk_sizes,
                       uint64_t expected_size) {
  if (compressed.empty()) {
    if (!chunk_sizes.empty() || expected_size != 0) {
      throw std::runtime_error(
          "Compressed chunk metadata is inconsistent for empty payload");
    }
    return {};
  }

  if (sum_sizes(chunk_sizes) != compressed.size()) {
    throw std::runtime_error(
        "Compressed chunk sizes do not match payload size");
  }

  std::vector<uint8_t> output;
  output.reserve(expected_size);

  size_t offset = 0;
  for (uint64_t chunk_size : chunk_sizes) {
    std::vector<uint8_t> slice(
        compressed.begin() + static_cast<std::ptrdiff_t>(offset),
        compressed.begin() + static_cast<std::ptrdiff_t>(offset + chunk_size));
    const size_t expected_chunk_size = static_cast<size_t>(std::min<uint64_t>(
        MAX_FIELD_SLICE_SIZE, expected_size - output.size()));
    ZstdCompressedBlock block{std::move(slice), expected_chunk_size};
    auto decompressed = nvcomp_zstd_decompress(block);
    output.insert(output.end(), decompressed.begin(), decompressed.end());
    offset += chunk_size;
  }

  if (offset != compressed.size() || output.size() != expected_size) {
    throw std::runtime_error(
        "Chunked decompression produced an unexpected size");
  }

  return output;
}

FastqData rebuild_fastq(const std::vector<uint64_t> &line_offsets,
                        const std::vector<uint8_t> &identifiers,
                        const std::vector<uint8_t> &basecalls,
                        const std::vector<uint8_t> &quality_scores,
                        uint64_t num_records) {
  FastqData data;
  data.line_offsets = line_offsets;
  data.num_records = num_records;

  if (line_offsets.empty()) {
    throw std::runtime_error("Decoded line-offset metadata is empty");
  }

  const uint64_t file_size = line_offsets.back();
  data.raw_bytes.resize(file_size);

  uint64_t id_offset = 0;
  uint64_t seq_offset = 0;
  uint64_t qual_offset = 0;

  for (uint64_t record = 0; record < num_records; ++record) {
    const uint64_t id_line = 4 * record;
    const uint64_t seq_line = id_line + 1;
    const uint64_t plus_line = id_line + 2;
    const uint64_t qual_line = id_line + 3;

    const uint64_t id_start = line_offsets[id_line];
    const uint64_t seq_start = line_offsets[seq_line];
    const uint64_t plus_start = line_offsets[plus_line];
    const uint64_t qual_start = line_offsets[qual_line];
    const uint64_t next_start = line_offsets[qual_line + 1];

    const uint64_t id_len = seq_start - id_start - 2;
    const uint64_t seq_len = plus_start - seq_start - 1;
    const uint64_t plus_len = qual_start - plus_start - 1;
    const uint64_t qual_len = next_start - qual_start - 1;

    if (plus_len != 1) {
      throw std::runtime_error("Decoded line-offset metadata is incompatible "
                               "with ignored plus lines");
    }
    if (id_offset + id_len > identifiers.size() ||
        seq_offset + seq_len > basecalls.size() ||
        qual_offset + qual_len > quality_scores.size()) {
      throw std::runtime_error(
          "Decoded FASTQ field stream exceeds its uncompressed size");
    }

    data.raw_bytes[id_start] = '@';
    std::memcpy(data.raw_bytes.data() + id_start + 1,
                identifiers.data() + id_offset, id_len);
    data.raw_bytes[seq_start - 1] = '\n';

    std::memcpy(data.raw_bytes.data() + seq_start,
                basecalls.data() + seq_offset, seq_len);
    data.raw_bytes[plus_start - 1] = '\n';

    data.raw_bytes[plus_start] = '+';
    data.raw_bytes[qual_start - 1] = '\n';

    std::memcpy(data.raw_bytes.data() + qual_start,
                quality_scores.data() + qual_offset, qual_len);
    data.raw_bytes[next_start - 1] = '\n';

    id_offset += id_len;
    seq_offset += seq_len;
    qual_offset += qual_len;
  }

  if (id_offset != identifiers.size() || seq_offset != basecalls.size() ||
      qual_offset != quality_scores.size()) {
    throw std::runtime_error(
        "Decoded FASTQ field streams contain trailing bytes outside the index");
  }

  return data;
}

} // namespace

std::vector<uint8_t> gpu_compress(const std::vector<uint8_t> &input,
                                  size_t chunk_size) {
  if (input.empty()) {
    return {};
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint8_t *d_input = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input.size()));
  CUDA_CHECK(cudaMemcpyAsync(d_input, input.data(), input.size(),
                             cudaMemcpyHostToDevice, stream));

  std::vector<uint8_t> output =
      nvcomp_zstd_compress_device(d_input, input.size(), chunk_size, stream)
          .payload;

  cudaFree(d_input);
  cudaStreamDestroy(stream);
  return output;
}

std::vector<uint8_t> gpu_decompress(const std::vector<uint8_t> &compressed) {
  if (compressed.empty()) {
    return {};
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<uint8_t> output =
      nvcomp_zstd_decompress(ZstdCompressedBlock{compressed, compressed.size()},
                             NVCOMP_ZSTD_CHUNK_SIZE_DEFAULT, stream);
  cudaStreamDestroy(stream);
  return output;
}

CompressedFastqData compress_fastq(const FastqData &data, size_t chunk_size) {
  const FastqFieldStats stats = compute_field_stats(data);
  const size_t field_slice_size = MAX_FIELD_SLICE_SIZE;

  CompressedFastqData result;
  result.num_records = data.num_records;
  result.identifiers.original_size = stats.identifiers_size;
  result.basecalls.original_size = stats.basecalls_size;
  result.quality_scores.original_size = stats.quality_scores_size;
  result.line_lengths.original_size = stats.line_length_size;
  result.line_offset_count = data.line_offsets.size();

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint8_t *d_raw_bytes = nullptr;
  uint64_t *d_line_offsets = nullptr;
  uint64_t *d_identifier_lengths = nullptr;
  uint64_t *d_basecall_lengths = nullptr;
  uint64_t *d_quality_lengths = nullptr;
  uint64_t *d_identifier_offsets = nullptr;
  uint64_t *d_basecall_offsets = nullptr;
  uint64_t *d_quality_offsets = nullptr;
  uint32_t *d_line_lengths = nullptr;
  DeviceFieldBuffers fields;

  try {
    if (!data.raw_bytes.empty()) {
      CUDA_CHECK(cudaMalloc(&d_raw_bytes, data.raw_bytes.size()));
      CUDA_CHECK(cudaMemcpyAsync(d_raw_bytes, data.raw_bytes.data(),
                                 data.raw_bytes.size(), cudaMemcpyHostToDevice,
                                 stream));
    }

    if (!data.line_offsets.empty()) {
      CUDA_CHECK(cudaMalloc(&d_line_offsets,
                            data.line_offsets.size() * sizeof(uint64_t)));
      CUDA_CHECK(cudaMemcpyAsync(d_line_offsets, data.line_offsets.data(),
                                 data.line_offsets.size() * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, stream));
    }

    if (data.num_records > 0) {
      CUDA_CHECK(cudaMalloc(&d_identifier_lengths,
                            data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(
          cudaMalloc(&d_basecall_lengths, data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(
          cudaMalloc(&d_quality_lengths, data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(cudaMalloc(&d_identifier_offsets,
                            data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(
          cudaMalloc(&d_basecall_offsets, data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(
          cudaMalloc(&d_quality_offsets, data.num_records * sizeof(uint64_t)));

      const uint32_t block_size = 256;
      const uint32_t grid_size = static_cast<uint32_t>(
          (data.num_records + block_size - 1) / block_size);

      compute_field_lengths_kernel<<<grid_size, block_size, 0, stream>>>(
          d_line_offsets, d_identifier_lengths, d_basecall_lengths,
          d_quality_lengths, data.num_records);
      CUDA_CHECK(cudaGetLastError());

      thrust::exclusive_scan(thrust::cuda::par.on(stream),
                             thrust::device_pointer_cast(d_identifier_lengths),
                             thrust::device_pointer_cast(d_identifier_lengths) +
                                 data.num_records,
                             thrust::device_pointer_cast(d_identifier_offsets));
      thrust::exclusive_scan(thrust::cuda::par.on(stream),
                             thrust::device_pointer_cast(d_basecall_lengths),
                             thrust::device_pointer_cast(d_basecall_lengths) +
                                 data.num_records,
                             thrust::device_pointer_cast(d_basecall_offsets));
      thrust::exclusive_scan(thrust::cuda::par.on(stream),
                             thrust::device_pointer_cast(d_quality_lengths),
                             thrust::device_pointer_cast(d_quality_lengths) +
                                 data.num_records,
                             thrust::device_pointer_cast(d_quality_offsets));

      if (stats.identifiers_size > 0) {
        CUDA_CHECK(cudaMalloc(&fields.identifiers, stats.identifiers_size));
      }
      if (stats.basecalls_size > 0) {
        CUDA_CHECK(cudaMalloc(&fields.basecalls, stats.basecalls_size));
      }
      if (stats.quality_scores_size > 0) {
        CUDA_CHECK(
            cudaMalloc(&fields.quality_scores, stats.quality_scores_size));
      }

      gather_fields_kernel<<<grid_size, block_size, 0, stream>>>(
          d_raw_bytes, d_line_offsets, d_identifier_offsets, d_basecall_offsets,
          d_quality_offsets, d_identifier_lengths, d_basecall_lengths,
          d_quality_lengths, fields.identifiers, fields.basecalls,
          fields.quality_scores, data.num_records);
      CUDA_CHECK(cudaGetLastError());
    }

    if (!data.line_offsets.empty()) {
      const uint64_t line_length_count = data.line_offsets.size();
      CUDA_CHECK(
          cudaMalloc(&d_line_lengths, line_length_count * sizeof(uint32_t)));
      delta_encode_offsets_to_lengths(d_line_offsets, d_line_lengths,
                                      line_length_count, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cerr << "Compressing identifiers (" << stats.identifiers_size
              << " bytes)..." << std::endl;
    auto id_chunks =
        gpu_compress_device_chunked(fields.identifiers, stats.identifiers_size,
                                    field_slice_size, chunk_size, stream);
    result.identifiers.payload = std::move(id_chunks.data);
    result.compressed_identifier_chunk_sizes = std::move(id_chunks.chunk_sizes);
    std::cerr << "  -> " << result.identifiers.payload.size() << " bytes"
              << std::endl;

    std::cerr << "Compressing basecalls (" << stats.basecalls_size
              << " bytes)..." << std::endl;
    auto seq_chunks =
        gpu_compress_device_chunked(fields.basecalls, stats.basecalls_size,
                                    field_slice_size, chunk_size, stream);
    result.basecalls.payload = std::move(seq_chunks.data);
    result.compressed_basecall_chunk_sizes = std::move(seq_chunks.chunk_sizes);
    std::cerr << "  -> " << result.basecalls.payload.size() << " bytes"
              << std::endl;

    std::cerr << "Compressing quality scores (" << stats.quality_scores_size
              << " bytes)..." << std::endl;
    std::vector<uint8_t> h_quality_scores(stats.quality_scores_size);
    if (stats.quality_scores_size > 0) {
      CUDA_CHECK(cudaMemcpyAsync(h_quality_scores.data(), fields.quality_scores,
                                 stats.quality_scores_size,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto qual_chunks = bsc_compress_host_chunked(
        h_quality_scores.data(), stats.quality_scores_size, 8 * 1024 * 1024);
    result.quality_scores.payload = std::move(qual_chunks.data);
    result.compressed_quality_chunk_sizes = std::move(qual_chunks.chunk_sizes);
    std::cerr << "  -> " << result.quality_scores.payload.size() << " bytes"
              << std::endl;

    std::cerr << "Compressing line lengths (" << stats.line_length_size
              << " bytes)..." << std::endl;
    auto index_chunks = gpu_compress_device_chunked(
        reinterpret_cast<const uint8_t *>(d_line_lengths),
        stats.line_length_size, field_slice_size, chunk_size, stream);
    result.line_lengths.payload = std::move(index_chunks.data);
    result.compressed_line_length_chunk_sizes =
        std::move(index_chunks.chunk_sizes);
    std::cerr << "  -> " << result.line_lengths.payload.size() << " bytes"
              << std::endl;
  } catch (...) {
    cuda_free_if_set(d_line_lengths);
    cuda_free_if_set(fields.identifiers);
    cuda_free_if_set(fields.basecalls);
    cuda_free_if_set(fields.quality_scores);
    cuda_free_if_set(d_quality_offsets);
    cuda_free_if_set(d_basecall_offsets);
    cuda_free_if_set(d_identifier_offsets);
    cuda_free_if_set(d_quality_lengths);
    cuda_free_if_set(d_basecall_lengths);
    cuda_free_if_set(d_identifier_lengths);
    cuda_free_if_set(d_line_offsets);
    cuda_free_if_set(d_raw_bytes);
    cudaStreamDestroy(stream);
    throw;
  }

  cuda_free_if_set(d_line_lengths);
  cuda_free_if_set(fields.identifiers);
  cuda_free_if_set(fields.basecalls);
  cuda_free_if_set(fields.quality_scores);
  cuda_free_if_set(d_quality_offsets);
  cuda_free_if_set(d_basecall_offsets);
  cuda_free_if_set(d_identifier_offsets);
  cuda_free_if_set(d_quality_lengths);
  cuda_free_if_set(d_basecall_lengths);
  cuda_free_if_set(d_identifier_lengths);
  cuda_free_if_set(d_line_offsets);
  cuda_free_if_set(d_raw_bytes);
  cudaStreamDestroy(stream);
  return result;
}

FastqData decompress_fastq(const CompressedFastqData &compressed) {
  std::cerr << "Decompressing identifiers..." << std::endl;
  const auto identifiers =
      gpu_decompress_chunked(compressed.identifiers.payload,
                             compressed.compressed_identifier_chunk_sizes,
                             compressed.identifiers.original_size);
  std::cerr << "  -> " << identifiers.size() << " bytes" << std::endl;

  std::cerr << "Decompressing basecalls..." << std::endl;
  const auto basecalls = gpu_decompress_chunked(
      compressed.basecalls.payload, compressed.compressed_basecall_chunk_sizes,
      compressed.basecalls.original_size);
  std::cerr << "  -> " << basecalls.size() << " bytes" << std::endl;

  std::cerr << "Decompressing quality scores..." << std::endl;
  const auto quality_scores =
      bsc_decompress_host_chunked(compressed.quality_scores.payload,
                                  compressed.compressed_quality_chunk_sizes,
                                  compressed.quality_scores.original_size);
  std::cerr << "  -> " << quality_scores.size() << " bytes" << std::endl;

  std::cerr << "Decompressing line lengths..." << std::endl;
  const auto line_offset_bytes =
      gpu_decompress_chunked(compressed.line_lengths.payload,
                             compressed.compressed_line_length_chunk_sizes,
                             compressed.line_lengths.original_size);
  std::cerr << "  -> " << line_offset_bytes.size() << " bytes" << std::endl;

  if (compressed.line_offset_count == 0) {
    throw std::runtime_error("Decoded line-offset count is invalid");
  }
  if (line_offset_bytes.size() !=
      compressed.line_offset_count * sizeof(uint32_t)) {
    throw std::runtime_error(
        "Decoded line-length payload has an unexpected size");
  }

  std::vector<uint64_t> line_offsets(compressed.line_offset_count);
  uint32_t *d_line_lengths = nullptr;
  uint64_t *d_line_offsets = nullptr;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  try {
    CUDA_CHECK(cudaMalloc(&d_line_lengths, line_offset_bytes.size()));
    CUDA_CHECK(
        cudaMalloc(&d_line_offsets, line_offsets.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_line_lengths, line_offset_bytes.data(),
                               line_offset_bytes.size(), cudaMemcpyHostToDevice,
                               stream));
    delta_decode_lengths_to_offsets(d_line_lengths, d_line_offsets,
                                    line_offsets.size(), stream);
    CUDA_CHECK(cudaMemcpyAsync(line_offsets.data(), d_line_offsets,
                               line_offsets.size() * sizeof(uint64_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } catch (...) {
    cuda_free_if_set(d_line_offsets);
    cuda_free_if_set(d_line_lengths);
    cudaStreamDestroy(stream);
    throw;
  }

  cuda_free_if_set(d_line_offsets);
  cuda_free_if_set(d_line_lengths);
  cudaStreamDestroy(stream);
  return rebuild_fastq(line_offsets, identifiers, basecalls, quality_scores,
                       compressed.num_records);
}

} // namespace gpufastq
