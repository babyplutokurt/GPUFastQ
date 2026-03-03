#include "gpufastq/codec_bsc.hpp"

#include <libbsc/libbsc.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace gpufastq {

namespace {

void bsc_check_init() {
  const int rc = bsc_init(LIBBSC_FEATURE_FASTMODE);
  if (rc != LIBBSC_NO_ERROR) {
    throw std::runtime_error("BSC initialization failed with code: " +
                             std::to_string(rc));
  }
}

} // namespace

BscChunkedBuffer bsc_compress_chunked(const uint8_t *input, size_t input_size,
                                      size_t chunk_size) {
  BscChunkedBuffer result;
  if (input_size == 0) {
    return result;
  }
  if (input == nullptr) {
    throw std::runtime_error("BSC compression input pointer is null");
  }
  if (chunk_size == 0) {
    throw std::runtime_error("BSC compression chunk size must be non-zero");
  }

  bsc_check_init();

  for (size_t offset = 0; offset < input_size; offset += chunk_size) {
    const size_t current_chunk_size = std::min(chunk_size, input_size - offset);
    if (current_chunk_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("BSC chunk exceeds supported size");
    }

    const int input_chunk_size = static_cast<int>(current_chunk_size);
    std::vector<uint8_t> compressed(current_chunk_size + LIBBSC_HEADER_SIZE);

    const int compressed_size =
        bsc_compress(input + offset, compressed.data(), input_chunk_size, 16,
                     128, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_ADAPTIVE,
                     LIBBSC_FEATURE_FASTMODE);
    if (compressed_size < LIBBSC_NO_ERROR) {
      throw std::runtime_error("BSC compression failed with code: " +
                               std::to_string(compressed_size));
    }

    result.compressed_chunk_sizes.push_back(compressed_size);
    result.uncompressed_chunk_sizes.push_back(current_chunk_size);
    result.data.insert(result.data.end(), compressed.begin(),
                       compressed.begin() + compressed_size);
  }

  return result;
}

std::vector<uint8_t>
bsc_decompress_chunked(const std::vector<uint8_t> &compressed,
                       const std::vector<uint64_t> &compressed_chunk_sizes,
                       const std::vector<uint64_t> &uncompressed_chunk_sizes,
                       uint64_t expected_size) {
  if (compressed.empty()) {
    if (!compressed_chunk_sizes.empty() || !uncompressed_chunk_sizes.empty() ||
        expected_size != 0) {
      throw std::runtime_error(
          "BSC chunk metadata is inconsistent for empty payload");
    }
    return {};
  }

  if (compressed_chunk_sizes.size() != uncompressed_chunk_sizes.size()) {
    throw std::runtime_error("BSC chunk metadata sizes do not match");
  }

  bsc_check_init();

  std::vector<uint8_t> output(expected_size);
  size_t compressed_offset = 0;
  size_t uncompressed_offset = 0;

  for (size_t i = 0; i < compressed_chunk_sizes.size(); ++i) {
    const uint64_t compressed_chunk_size = compressed_chunk_sizes[i];
    const uint64_t uncompressed_chunk_size = uncompressed_chunk_sizes[i];

    if (compressed_chunk_size >
            static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        uncompressed_chunk_size >
            static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("BSC chunk exceeds supported size");
    }
    if (compressed_offset + compressed_chunk_size > compressed.size()) {
      throw std::runtime_error("BSC compressed chunk metadata exceeds payload");
    }
    if (uncompressed_offset + uncompressed_chunk_size > output.size()) {
      throw std::runtime_error("BSC uncompressed chunk metadata exceeds output");
    }

    int block_size = 0;
    int data_size = 0;
    const int info_result = bsc_block_info(
        compressed.data() + compressed_offset,
        static_cast<int>(compressed_chunk_size), &block_size, &data_size,
        LIBBSC_FEATURE_FASTMODE);
    if (info_result != LIBBSC_NO_ERROR) {
      throw std::runtime_error("BSC block info failed with code: " +
                               std::to_string(info_result));
    }
    if (static_cast<uint64_t>(block_size) != compressed_chunk_size ||
        static_cast<uint64_t>(data_size) != uncompressed_chunk_size) {
      throw std::runtime_error("BSC chunk metadata does not match block header");
    }

    const int decomp_result = bsc_decompress(
        compressed.data() + compressed_offset,
        static_cast<int>(compressed_chunk_size),
        output.data() + uncompressed_offset,
        static_cast<int>(uncompressed_chunk_size), LIBBSC_FEATURE_FASTMODE);
    if (decomp_result != LIBBSC_NO_ERROR) {
      throw std::runtime_error("BSC decompression failed with code: " +
                               std::to_string(decomp_result));
    }

    compressed_offset += compressed_chunk_size;
    uncompressed_offset += uncompressed_chunk_size;
  }

  if (compressed_offset != compressed.size() ||
      uncompressed_offset != expected_size) {
    throw std::runtime_error("BSC chunked decompression produced an unexpected size");
  }

  return output;
}

} // namespace gpufastq
