#pragma once

#include "fastq_record.hpp"
#include <cstdint>
#include <vector>

namespace gpufastq {

/// Default chunk size for nvcomp zstd (64KB recommended for best performance)
constexpr size_t DEFAULT_CHUNK_SIZE = 1 << 16; // 65536

/// Compress a single buffer using GPU-accelerated zstd (nvcomp)
std::vector<uint8_t> gpu_compress(const std::vector<uint8_t> &input,
                                  size_t chunk_size = DEFAULT_CHUNK_SIZE);

/// Decompress a single buffer using GPU-accelerated zstd (nvcomp)
std::vector<uint8_t> gpu_decompress(const std::vector<uint8_t> &compressed);

/// Compress all three FASTQ fields on GPU
CompressedFastqData compress_fastq(const FastqData &data,
                                   size_t chunk_size = DEFAULT_CHUNK_SIZE);

/// Decompress all three FASTQ fields on GPU
FastqData decompress_fastq(const CompressedFastqData &compressed);

} // namespace gpufastq
