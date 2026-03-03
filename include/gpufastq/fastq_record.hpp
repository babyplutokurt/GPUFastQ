#pragma once

#include <cstdint>
#include <vector>

namespace gpufastq {

/// Raw FASTQ data split into three field vectors
struct FastqData {
  /// All identifier lines (without '@' prefix), separated by '\n'
  std::vector<uint8_t> identifiers;
  /// All basecall/sequence lines, separated by '\n'
  std::vector<uint8_t> basecalls;
  /// All quality score lines, separated by '\n'
  std::vector<uint8_t> quality_scores;
  /// Number of records
  uint64_t num_records = 0;
};

/// Compressed FASTQ data with metadata for decompression
struct CompressedFastqData {
  std::vector<uint8_t> compressed_identifiers;
  std::vector<uint8_t> compressed_basecalls;
  std::vector<uint8_t> compressed_quality;

  uint64_t original_id_size = 0;
  uint64_t original_seq_size = 0;
  uint64_t original_qual_size = 0;
  uint64_t num_records = 0;
};

} // namespace gpufastq
