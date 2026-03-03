#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpufastq {

enum class IdentifierColumnKind : uint8_t {
  String = 0,
  Int32 = 1,
};

enum class IdentifierColumnEncoding : uint8_t {
  Plain = 0,
  Delta = 1,
};

enum class IdentifierCompressionMode : uint8_t {
  Flat = 0,
  Columnar = 1,
};

struct IdentifierLayout {
  bool columnar = false;
  std::vector<std::string> separators;
  std::vector<IdentifierColumnKind> column_kinds;
};

/// Raw FASTQ bytes plus line-start indices.
struct FastqData {
  /// Full FASTQ byte stream.
  std::vector<uint8_t> raw_bytes;
  /// Start offset of each FASTQ line. The final entry is a sentinel: file_size.
  std::vector<uint64_t> line_offsets;
  /// Number of four-line FASTQ records.
  uint64_t num_records = 0;
  /// Discovered identifier layout from the initial FASTQ sample.
  IdentifierLayout identifier_layout;
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

struct CompressedIdentifierColumn {
  IdentifierColumnKind kind = IdentifierColumnKind::String;
  IdentifierColumnEncoding encoding = IdentifierColumnEncoding::Plain;
  ZstdCompressedBlock values;
  ZstdCompressedBlock lengths;
  std::vector<uint64_t> compressed_value_chunk_sizes;
  std::vector<uint64_t> compressed_length_chunk_sizes;
};

struct CompressedIdentifierData {
  IdentifierCompressionMode mode = IdentifierCompressionMode::Flat;
  uint64_t original_size = 0;
  IdentifierLayout layout;
  ZstdCompressedBlock flat_data;
  std::vector<uint64_t> compressed_flat_chunk_sizes;
  std::vector<CompressedIdentifierColumn> columns;
};

struct CompressedBasecallData {
  uint64_t original_size = 0;
  uint32_t n_block_size = 0;
  ZstdCompressedBlock packed_bases;
  ZstdCompressedBlock n_counts;
  ZstdCompressedBlock n_positions;
  std::vector<uint64_t> compressed_packed_chunk_sizes;
  std::vector<uint64_t> compressed_n_position_chunk_sizes;
};

/// Compressed FASTQ field streams plus compressed line-length metadata.
struct CompressedFastqData {
  CompressedIdentifierData identifiers;
  CompressedBasecallData basecalls;
  ZstdCompressedBlock quality_scores;
  ZstdCompressedBlock line_lengths;

  std::vector<uint64_t> compressed_quality_chunk_sizes;
  std::vector<uint64_t> uncompressed_quality_chunk_sizes;
  std::vector<uint64_t> compressed_line_length_chunk_sizes;

  uint64_t line_offset_count = 0;
  uint64_t num_records = 0;
};

} // namespace gpufastq
