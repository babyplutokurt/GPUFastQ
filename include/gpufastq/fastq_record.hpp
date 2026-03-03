#pragma once

#include <cstdint>
#include <vector>

namespace gpufastq {

/// Raw FASTQ data split into three field vectors
struct FastqData {
  /// All identifier lines (without '@' prefix), concatenated record by record.
  std::vector<uint8_t> identifiers;
  /// All basecall/sequence lines, concatenated record by record.
  std::vector<uint8_t> basecalls;
  /// All quality score lines, concatenated record by record.
  std::vector<uint8_t> quality_scores;
  /// Per-record field lengths used as the reconstruction index.
  std::vector<uint64_t> identifier_lengths;
  std::vector<uint64_t> basecall_lengths;
  std::vector<uint64_t> quality_lengths;
  /// Number of records
  uint64_t num_records = 0;
};

/// Compressed FASTQ data with metadata for decompression
struct CompressedFastqData {
  std::vector<uint8_t> compressed_identifiers;
  std::vector<uint8_t> compressed_basecalls;
  std::vector<uint8_t> compressed_quality;
  std::vector<uint64_t> compressed_identifier_chunk_sizes;
  std::vector<uint64_t> compressed_basecall_chunk_sizes;
  std::vector<uint64_t> compressed_quality_chunk_sizes;

  std::vector<uint64_t> identifier_lengths;
  std::vector<uint64_t> basecall_lengths;
  std::vector<uint64_t> quality_lengths;

  uint64_t original_id_size = 0;
  uint64_t original_seq_size = 0;
  uint64_t original_qual_size = 0;
  uint64_t num_records = 0;
};

} // namespace gpufastq
