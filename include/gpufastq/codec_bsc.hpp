#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpufastq {

constexpr size_t BSC_QUALITY_CHUNK_SIZE = 8 * 1024 * 1024;

struct BscChunkedBuffer {
  std::vector<uint8_t> data;
  std::vector<uint64_t> compressed_chunk_sizes;
  std::vector<uint64_t> uncompressed_chunk_sizes;
};

BscChunkedBuffer bsc_compress_chunked(const uint8_t *input, size_t input_size,
                                      size_t chunk_size = BSC_QUALITY_CHUNK_SIZE);

std::vector<uint8_t>
bsc_decompress_chunked(const std::vector<uint8_t> &compressed,
                       const std::vector<uint64_t> &compressed_chunk_sizes,
                       const std::vector<uint64_t> &uncompressed_chunk_sizes,
                       uint64_t expected_size);

} // namespace gpufastq
