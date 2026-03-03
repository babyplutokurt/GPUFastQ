#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpufastq {

/// Raw FASTQ bytes plus line-start indices.
struct FastqData {
  /// Full FASTQ byte stream.
  std::vector<uint8_t> raw_bytes;
  /// Start offset of each FASTQ line. The final entry is a sentinel: file_size.
  std::vector<uint64_t> line_offsets;
  /// Number of four-line FASTQ records.
  uint64_t num_records = 0;
};

struct FastqFieldStats {
  uint64_t identifiers_size = 0;
  uint64_t basecalls_size = 0;
  uint64_t quality_scores_size = 0;
  uint64_t line_length_size = 0;
};

struct ZstdCompressedBlock {
  std::vector<uint8_t> payload;
  size_t original_size = 0;
};

/// Compressed FASTQ field streams plus compressed line-length metadata.
struct CompressedFastqData {
  ZstdCompressedBlock identifiers;
  ZstdCompressedBlock basecalls;
  ZstdCompressedBlock quality_scores;
  ZstdCompressedBlock line_lengths;

  std::vector<uint64_t> compressed_identifier_chunk_sizes;
  std::vector<uint64_t> compressed_basecall_chunk_sizes;
  std::vector<uint64_t> compressed_quality_chunk_sizes;
  std::vector<uint64_t> compressed_line_length_chunk_sizes;

  uint64_t line_offset_count = 0;
  uint64_t num_records = 0;
};

} // namespace gpufastq
