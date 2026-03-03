#include "gpufastq/fastq_parser.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace gpufastq {

FastqData parse_fastq(const std::string &filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open FASTQ file: " + filepath);
  }

  FastqData data;
  std::string line;
  uint64_t line_num = 0;

  while (std::getline(file, line)) {
    // Handle Windows line endings
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }

    switch (line_num % 4) {
    case 0: {
      // Identifier line (starts with @)
      if (line.empty() || line[0] != '@') {
        throw std::runtime_error("Expected '@' at line " +
                                 std::to_string(line_num + 1));
      }
      // Store without '@' prefix
      auto id = line.substr(1);
      data.identifiers.insert(data.identifiers.end(), id.begin(), id.end());
      data.identifier_lengths.push_back(id.size());
      break;
    }
    case 1: {
      // Basecall / sequence line
      data.basecalls.insert(data.basecalls.end(), line.begin(), line.end());
      data.basecall_lengths.push_back(line.size());
      break;
    }
    case 2: {
      // '+' separator — skip
      if (line.empty() || line[0] != '+') {
        throw std::runtime_error("Expected '+' at line " +
                                 std::to_string(line_num + 1));
      }
      break;
    }
    case 3: {
      // Quality score line
      if (data.basecall_lengths.empty()) {
        throw std::runtime_error("Missing sequence before quality line " +
                                 std::to_string(line_num + 1));
      }
      if (line.size() != data.basecall_lengths.back()) {
        throw std::runtime_error("Sequence/quality length mismatch at record " +
                                 std::to_string(data.num_records + 1));
      }
      data.quality_scores.insert(data.quality_scores.end(), line.begin(),
                                 line.end());
      data.quality_lengths.push_back(line.size());
      data.num_records++;
      break;
    }
    }
    line_num++;
  }

  if (line_num % 4 != 0) {
    throw std::runtime_error("Incomplete FASTQ record at end of file");
  }

  return data;
}

void write_fastq(const std::string &filepath, const FastqData &data) {
  std::ofstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open output file: " + filepath);
  }

  if (data.identifier_lengths.size() != data.num_records ||
      data.basecall_lengths.size() != data.num_records ||
      data.quality_lengths.size() != data.num_records) {
    throw std::runtime_error(
        "Record count mismatch during FASTQ reconstruction");
  }

  uint64_t id_offset = 0;
  uint64_t seq_offset = 0;
  uint64_t qual_offset = 0;

  for (uint64_t i = 0; i < data.num_records; i++) {
    const auto id_len = data.identifier_lengths[i];
    const auto seq_len = data.basecall_lengths[i];
    const auto qual_len = data.quality_lengths[i];

    if (qual_len != seq_len) {
      throw std::runtime_error("Sequence/quality length mismatch during FASTQ reconstruction");
    }
    if (id_offset + id_len > data.identifiers.size() ||
        seq_offset + seq_len > data.basecalls.size() ||
        qual_offset + qual_len > data.quality_scores.size()) {
      throw std::runtime_error("Index exceeds FASTQ field buffer size");
    }

    file << '@';
    file.write(reinterpret_cast<const char *>(data.identifiers.data() + id_offset),
               static_cast<std::streamsize>(id_len));
    file << '\n';
    file.write(reinterpret_cast<const char *>(data.basecalls.data() + seq_offset),
               static_cast<std::streamsize>(seq_len));
    file << '\n'
         << '+'
         << '\n';
    file.write(reinterpret_cast<const char *>(data.quality_scores.data() + qual_offset),
               static_cast<std::streamsize>(qual_len));
    file << '\n';

    id_offset += id_len;
    seq_offset += seq_len;
    qual_offset += qual_len;
  }

  if (id_offset != data.identifiers.size() ||
      seq_offset != data.basecalls.size() ||
      qual_offset != data.quality_scores.size()) {
    throw std::runtime_error("FASTQ field buffer contains trailing data outside the index");
  }
}

} // namespace gpufastq
