#include "gpufastq/gpu_compressor.hpp"

#include <cuda_runtime.h>
#include <nvcomp/zstd.hpp>

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace gpufastq {

namespace {

constexpr size_t MAX_FIELD_SLICE_SIZE = 16 * 1024 * 1024;

struct ChunkedCompressedBuffer {
  std::vector<uint8_t> data;
  std::vector<uint64_t> chunk_sizes;
};

uint64_t sum_sizes(const std::vector<uint64_t> &sizes) {
  return std::accumulate(sizes.begin(), sizes.end(), uint64_t{0});
}

ChunkedCompressedBuffer gpu_compress_chunked(const std::vector<uint8_t> &input,
                                             size_t field_slice_size,
                                             size_t nvcomp_chunk_size) {
  ChunkedCompressedBuffer result;
  if (input.empty()) {
    return result;
  }

  if (field_slice_size == 0 || field_slice_size > MAX_FIELD_SLICE_SIZE) {
    throw std::runtime_error(
        "Requested field slice size is out of supported range");
  }
  if (nvcomp_chunk_size == 0 || nvcomp_chunk_size > MAX_FIELD_SLICE_SIZE) {
    throw std::runtime_error(
        "Requested nvcomp chunk size is out of supported range");
  }

  for (size_t offset = 0; offset < input.size(); offset += field_slice_size) {
    const size_t slice_size = std::min(field_slice_size, input.size() - offset);
    std::vector<uint8_t> slice(
        input.begin() + static_cast<std::ptrdiff_t>(offset),
        input.begin() + static_cast<std::ptrdiff_t>(offset + slice_size));
    auto compressed = gpu_compress(slice, nvcomp_chunk_size);
    result.chunk_sizes.push_back(compressed.size());
    result.data.insert(result.data.end(), compressed.begin(), compressed.end());
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
    auto decompressed = gpu_decompress(slice);
    output.insert(output.end(), decompressed.begin(), decompressed.end());
    offset += chunk_size;
  }

  if (offset != compressed.size() || output.size() != expected_size) {
    throw std::runtime_error(
        "Chunked decompression produced an unexpected size");
  }

  return output;
}

} // namespace

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +      \
                               ":" + std::to_string(__LINE__) + ": " +         \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

// ----------------------------------------------------------------
// Single-buffer compress / decompress using nvcomp ZstdManager
// ----------------------------------------------------------------

std::vector<uint8_t> gpu_compress(const std::vector<uint8_t> &input,
                                  size_t chunk_size) {
  if (input.empty())
    return {};

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<uint8_t> output;
  {
    nvcomp::ZstdManager manager(
        chunk_size, nvcompBatchedZstdCompressDefaultOpts,
        nvcompBatchedZstdDecompressDefaultOpts, stream,
        nvcomp::NoComputeNoVerify, nvcomp::BitstreamKind::NVCOMP_NATIVE);

    // Upload input to device
    uint8_t *d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_input, input.data(), input.size(),
                               cudaMemcpyHostToDevice, stream));

    // Configure compression
    auto comp_config = manager.configure_compression(input.size());

    // Allocate output buffer
    uint8_t *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, comp_config.max_compressed_buffer_size));

    // Compress
    size_t *d_comp_size = nullptr;
    CUDA_CHECK(cudaMalloc(&d_comp_size, sizeof(size_t)));
    manager.compress(d_input, d_output, comp_config, d_comp_size);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t comp_size = 0;
    CUDA_CHECK(cudaMemcpy(&comp_size, d_comp_size, sizeof(size_t),
                          cudaMemcpyDeviceToHost));

    // Copy back to host
    output.resize(comp_size);
    CUDA_CHECK(
        cudaMemcpy(output.data(), d_output, comp_size, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_comp_size);
  }
  cudaStreamDestroy(stream);
  return output;
}

std::vector<uint8_t> gpu_decompress(const std::vector<uint8_t> &compressed) {
  if (compressed.empty())
    return {};

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  nvcomp::ZstdManager manager(
      DEFAULT_CHUNK_SIZE, nvcompBatchedZstdCompressDefaultOpts,
      nvcompBatchedZstdDecompressDefaultOpts, stream, nvcomp::NoComputeNoVerify,
      nvcomp::BitstreamKind::NVCOMP_NATIVE);

  // Upload compressed data to device
  uint8_t *d_comp = nullptr;
  CUDA_CHECK(cudaMalloc(&d_comp, compressed.size()));
  CUDA_CHECK(cudaMemcpyAsync(d_comp, compressed.data(), compressed.size(),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Configure decompression (reads header from compressed data)
  auto decomp_config = manager.configure_decompression(d_comp);

  // Allocate output buffer
  uint8_t *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_output, decomp_config.decomp_data_size));

  // Decompress
  manager.decompress(d_output, d_comp, decomp_config);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Copy back to host
  std::vector<uint8_t> output(decomp_config.decomp_data_size);
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, decomp_config.decomp_data_size,
                        cudaMemcpyDeviceToHost));

  cudaFree(d_comp);
  cudaFree(d_output);
  cudaStreamDestroy(stream);
  return output;
}

// ----------------------------------------------------------------
// Compress / decompress all three FASTQ fields
// ----------------------------------------------------------------

CompressedFastqData compress_fastq(const FastqData &data, size_t chunk_size) {
  const size_t field_slice_size = MAX_FIELD_SLICE_SIZE;
  CompressedFastqData result;
  result.num_records = data.num_records;
  result.original_id_size = data.identifiers.size();
  result.original_seq_size = data.basecalls.size();
  result.original_qual_size = data.quality_scores.size();
  result.identifier_lengths = data.identifier_lengths;
  result.basecall_lengths = data.basecall_lengths;
  result.quality_lengths = data.quality_lengths;

  std::cerr << "Compressing identifiers (" << data.identifiers.size()
            << " bytes)..." << std::endl;
  auto id_chunks =
      gpu_compress_chunked(data.identifiers, field_slice_size, chunk_size);
  result.compressed_identifiers = std::move(id_chunks.data);
  result.compressed_identifier_chunk_sizes = std::move(id_chunks.chunk_sizes);
  std::cerr << "  -> " << result.compressed_identifiers.size() << " bytes"
            << std::endl;

  std::cerr << "Compressing basecalls (" << data.basecalls.size()
            << " bytes)..." << std::endl;
  auto seq_chunks =
      gpu_compress_chunked(data.basecalls, field_slice_size, chunk_size);
  result.compressed_basecalls = std::move(seq_chunks.data);
  result.compressed_basecall_chunk_sizes = std::move(seq_chunks.chunk_sizes);
  std::cerr << "  -> " << result.compressed_basecalls.size() << " bytes"
            << std::endl;

  std::cerr << "Compressing quality scores (" << data.quality_scores.size()
            << " bytes)..." << std::endl;
  auto qual_chunks =
      gpu_compress_chunked(data.quality_scores, field_slice_size, chunk_size);
  result.compressed_quality = std::move(qual_chunks.data);
  result.compressed_quality_chunk_sizes = std::move(qual_chunks.chunk_sizes);
  std::cerr << "  -> " << result.compressed_quality.size() << " bytes"
            << std::endl;

  return result;
}

FastqData decompress_fastq(const CompressedFastqData &compressed) {
  FastqData result;
  result.num_records = compressed.num_records;
  result.identifier_lengths = compressed.identifier_lengths;
  result.basecall_lengths = compressed.basecall_lengths;
  result.quality_lengths = compressed.quality_lengths;

  std::cerr << "Decompressing identifiers..." << std::endl;
  result.identifiers =
      gpu_decompress_chunked(compressed.compressed_identifiers,
                             compressed.compressed_identifier_chunk_sizes,
                             compressed.original_id_size);
  std::cerr << "  -> " << result.identifiers.size() << " bytes" << std::endl;

  std::cerr << "Decompressing basecalls..." << std::endl;
  result.basecalls = gpu_decompress_chunked(
      compressed.compressed_basecalls,
      compressed.compressed_basecall_chunk_sizes, compressed.original_seq_size);
  std::cerr << "  -> " << result.basecalls.size() << " bytes" << std::endl;

  std::cerr << "Decompressing quality scores..." << std::endl;
  result.quality_scores = gpu_decompress_chunked(
      compressed.compressed_quality, compressed.compressed_quality_chunk_sizes,
      compressed.original_qual_size);
  std::cerr << "  -> " << result.quality_scores.size() << " bytes" << std::endl;

  if (result.identifiers.size() != compressed.original_id_size ||
      result.basecalls.size() != compressed.original_seq_size ||
      result.quality_scores.size() != compressed.original_qual_size) {
    throw std::runtime_error(
        "Decompressed payload size does not match file metadata");
  }
  if (result.identifier_lengths.size() != result.num_records ||
      result.basecall_lengths.size() != result.num_records ||
      result.quality_lengths.size() != result.num_records) {
    throw std::runtime_error(
        "Decompressed FASTQ index metadata is inconsistent");
  }

  return result;
}

} // namespace gpufastq
