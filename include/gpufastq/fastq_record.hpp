#pragma once

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
  uint64_t line_index_size = 0;
};

/// Compressed FASTQ field streams plus compressed line-offset metadata.
struct CompressedFastqData {
  std::vector<uint8_t> compressed_identifiers;
  std::vector<uint8_t> compressed_basecalls;
  std::vector<uint8_t> compressed_quality;
  std::vector<uint8_t> compressed_line_offsets;

  std::vector<uint64_t> compressed_identifier_chunk_sizes;
  std::vector<uint64_t> compressed_basecall_chunk_sizes;
  std::vector<uint64_t> compressed_quality_chunk_sizes;
  std::vector<uint64_t> compressed_line_offset_chunk_sizes;

  uint64_t original_id_size = 0;
  uint64_t original_seq_size = 0;
  uint64_t original_qual_size = 0;
  uint64_t original_index_size = 0;
  uint64_t line_offset_count = 0;
  uint64_t num_records = 0;
};

} // namespace gpufastq
