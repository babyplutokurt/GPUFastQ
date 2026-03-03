#include "gpufastq/gpu_compressor.hpp"

#include <cuda_runtime.h>
#include <nvcomp/zstd.hpp>

#include <iostream>
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

// ----------------------------------------------------------------
// Single-buffer compress / decompress using nvcomp ZstdManager
// ----------------------------------------------------------------

std::vector<uint8_t> gpu_compress(const std::vector<uint8_t> &input,
                                  size_t chunk_size) {
  if (input.empty())
    return {};

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  nvcomp::ZstdManager manager(chunk_size, nvcompBatchedZstdCompressDefaultOpts,
                              nvcompBatchedZstdDecompressDefaultOpts, stream);

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
  manager.compress(d_input, d_output, comp_config);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Check status
  nvcompStatus_t *status = comp_config.get_status();
  if (status && *status != nvcompSuccess) {
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    throw std::runtime_error("nvcomp compression failed, status=" +
                             std::to_string(*status));
  }

  // Get actual compressed size
  size_t comp_size = manager.get_compressed_output_size(d_output);

  // Copy back to host
  std::vector<uint8_t> output(comp_size);
  CUDA_CHECK(
      cudaMemcpy(output.data(), d_output, comp_size, cudaMemcpyDeviceToHost));

  cudaFree(d_input);
  cudaFree(d_output);
  cudaStreamDestroy(stream);
  return output;
}

std::vector<uint8_t> gpu_decompress(const std::vector<uint8_t> &compressed) {
  if (compressed.empty())
    return {};

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  nvcomp::ZstdManager manager(DEFAULT_CHUNK_SIZE,
                              nvcompBatchedZstdCompressDefaultOpts,
                              nvcompBatchedZstdDecompressDefaultOpts, stream);

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

  // Check status
  nvcompStatus_t *status = decomp_config.get_status();
  if (status && *status != nvcompSuccess) {
    cudaFree(d_comp);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    throw std::runtime_error("nvcomp decompression failed, status=" +
                             std::to_string(*status));
  }

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
  CompressedFastqData result;
  result.num_records = data.num_records;
  result.original_id_size = data.identifiers.size();
  result.original_seq_size = data.basecalls.size();
  result.original_qual_size = data.quality_scores.size();

  std::cerr << "Compressing identifiers (" << data.identifiers.size()
            << " bytes)..." << std::endl;
  result.compressed_identifiers = gpu_compress(data.identifiers, chunk_size);
  std::cerr << "  -> " << result.compressed_identifiers.size() << " bytes"
            << std::endl;

  std::cerr << "Compressing basecalls (" << data.basecalls.size()
            << " bytes)..." << std::endl;
  result.compressed_basecalls = gpu_compress(data.basecalls, chunk_size);
  std::cerr << "  -> " << result.compressed_basecalls.size() << " bytes"
            << std::endl;

  std::cerr << "Compressing quality scores (" << data.quality_scores.size()
            << " bytes)..." << std::endl;
  result.compressed_quality = gpu_compress(data.quality_scores, chunk_size);
  std::cerr << "  -> " << result.compressed_quality.size() << " bytes"
            << std::endl;

  return result;
}

FastqData decompress_fastq(const CompressedFastqData &compressed) {
  FastqData result;
  result.num_records = compressed.num_records;

  std::cerr << "Decompressing identifiers..." << std::endl;
  result.identifiers = gpu_decompress(compressed.compressed_identifiers);
  std::cerr << "  -> " << result.identifiers.size() << " bytes" << std::endl;

  std::cerr << "Decompressing basecalls..." << std::endl;
  result.basecalls = gpu_decompress(compressed.compressed_basecalls);
  std::cerr << "  -> " << result.basecalls.size() << " bytes" << std::endl;

  std::cerr << "Decompressing quality scores..." << std::endl;
  result.quality_scores = gpu_decompress(compressed.compressed_quality);
  std::cerr << "  -> " << result.quality_scores.size() << " bytes" << std::endl;

  return result;
}

} // namespace gpufastq
