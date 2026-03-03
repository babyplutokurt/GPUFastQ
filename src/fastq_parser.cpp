#include "gpufastq/fastq_parser.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace gpufastq {

namespace {

uint64_t line_content_length(const std::vector<uint64_t> &line_offsets,
                             uint64_t line_idx) {
  const uint64_t line_start = line_offsets[line_idx];
  const uint64_t next_line_start = line_offsets[line_idx + 1];
  if (next_line_start <= line_start) {
    throw std::runtime_error("FASTQ line offsets are not strictly increasing");
  }
  return next_line_start - line_start - 1;
}

void validate_fastq_layout(const FastqData &data) {
  if (data.raw_bytes.empty()) {
    if (data.line_offsets.size() != 1 || data.line_offsets[0] != 0 ||
        data.num_records != 0) {
      throw std::runtime_error("Empty FASTQ metadata is inconsistent");
    }
    return;
  }

  if (data.raw_bytes.back() != '\n') {
    throw std::runtime_error("FASTQ file must end with a newline");
  }
  if (data.line_offsets.empty() || data.line_offsets.front() != 0) {
    throw std::runtime_error("FASTQ line index must start at byte offset 0");
  }
  if (data.line_offsets.back() != data.raw_bytes.size()) {
    throw std::runtime_error("FASTQ line index sentinel is inconsistent with the file size");
  }

  const uint64_t num_lines = data.line_offsets.size() - 1;
  if (num_lines % 4 != 0) {
    throw std::runtime_error("FASTQ file does not contain a multiple of 4 lines");
  }
  if (data.num_records != num_lines / 4) {
    throw std::runtime_error("FASTQ record count does not match the line index");
  }

  for (uint64_t record = 0; record < data.num_records; ++record) {
    const uint64_t id_line = 4 * record;
    const uint64_t seq_line = id_line + 1;
    const uint64_t plus_line = id_line + 2;
    const uint64_t qual_line = id_line + 3;

    const uint64_t id_start = data.line_offsets[id_line];
    const uint64_t seq_start = data.line_offsets[seq_line];
    const uint64_t plus_start = data.line_offsets[plus_line];
    const uint64_t qual_start = data.line_offsets[qual_line];

    if (data.raw_bytes[id_start] != '@') {
      throw std::runtime_error("Expected '@' at record " +
                               std::to_string(record + 1));
    }
    if (data.raw_bytes[plus_start] != '+') {
      throw std::runtime_error("Expected '+' at record " +
                               std::to_string(record + 1));
    }

    const uint64_t id_len = line_content_length(data.line_offsets, id_line);
    const uint64_t seq_len = line_content_length(data.line_offsets, seq_line);
    const uint64_t plus_len = line_content_length(data.line_offsets, plus_line);
    const uint64_t qual_len = line_content_length(data.line_offsets, qual_line);

    if (id_len <= 1) {
      throw std::runtime_error("Identifier line is empty at record " +
                               std::to_string(record + 1));
    }
    if (plus_len != 1) {
      throw std::runtime_error(
          "Only '+' separator lines without comments are supported");
    }
    if (seq_len != qual_len) {
      throw std::runtime_error("Sequence/quality length mismatch at record " +
                               std::to_string(record + 1));
    }
    if (seq_start != id_start + id_len + 1) {
      throw std::runtime_error("Identifier line length does not match line index");
    }
    if (plus_start != seq_start + seq_len + 1) {
      throw std::runtime_error("Sequence line length does not match line index");
    }
    if (qual_start != plus_start + plus_len + 1) {
      throw std::runtime_error("Plus line length does not match line index");
    }
  }
}

} // namespace

FastqData parse_fastq(const std::string &filepath) {
  std::ifstream file(filepath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open FASTQ file: " + filepath);
  }

  const auto end_pos = file.tellg();
  if (end_pos < 0) {
    throw std::runtime_error("Cannot determine FASTQ file size: " + filepath);
  }

  FastqData data;
  data.raw_bytes.resize(static_cast<size_t>(end_pos));

  file.seekg(0, std::ios::beg);
  if (!data.raw_bytes.empty()) {
    file.read(reinterpret_cast<char *>(data.raw_bytes.data()),
              static_cast<std::streamsize>(data.raw_bytes.size()));
    if (!file) {
      throw std::runtime_error("Failed to read FASTQ file: " + filepath);
    }
  }

  if (data.raw_bytes.empty()) {
    data.line_offsets = {0};
    return data;
  }

  data.line_offsets.reserve(data.raw_bytes.size() / 40 + 2);
  data.line_offsets.push_back(0);
  for (uint64_t i = 0; i < data.raw_bytes.size(); ++i) {
    if (data.raw_bytes[i] == '\n' && (i + 1) < data.raw_bytes.size()) {
      data.line_offsets.push_back(i + 1);
    }
  }
  data.line_offsets.push_back(data.raw_bytes.size());
  data.num_records = (data.line_offsets.size() - 1) / 4;

  validate_fastq_layout(data);
  return data;
}

FastqFieldStats compute_field_stats(const FastqData &data) {
  validate_fastq_layout(data);

  FastqFieldStats stats;
  stats.line_index_size = data.line_offsets.size() * sizeof(uint64_t);
  for (uint64_t record = 0; record < data.num_records; ++record) {
    const uint64_t id_line = 4 * record;
    const uint64_t seq_line = id_line + 1;
    const uint64_t qual_line = id_line + 3;

    stats.identifiers_size += line_content_length(data.line_offsets, id_line) - 1;
    stats.basecalls_size += line_content_length(data.line_offsets, seq_line);
    stats.quality_scores_size += line_content_length(data.line_offsets, qual_line);
  }

  return stats;
}

void write_fastq(const std::string &filepath, const FastqData &data) {
  std::ofstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open output file: " + filepath);
  }

  if (!data.raw_bytes.empty()) {
    file.write(reinterpret_cast<const char *>(data.raw_bytes.data()),
               static_cast<std::streamsize>(data.raw_bytes.size()));
    if (!file) {
      throw std::runtime_error("Failed to write FASTQ file: " + filepath);
    }
  }
}

} // namespace gpufastq
