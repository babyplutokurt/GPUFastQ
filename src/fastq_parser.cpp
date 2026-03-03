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
      // Newline separator between records
      if (data.num_records > 0) {
        data.identifiers.push_back('\n');
      }
      // Store without '@' prefix
      auto id = line.substr(1);
      data.identifiers.insert(data.identifiers.end(), id.begin(), id.end());
      break;
    }
    case 1: {
      // Basecall / sequence line
      if (data.num_records > 0) {
        data.basecalls.push_back('\n');
      }
      data.basecalls.insert(data.basecalls.end(), line.begin(), line.end());
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
      if (data.num_records > 0) {
        data.quality_scores.push_back('\n');
      }
      data.quality_scores.insert(data.quality_scores.end(), line.begin(),
                                 line.end());
      data.num_records++;
      break;
    }
    }
    line_num++;
  }

  return data;
}

void write_fastq(const std::string &filepath, const FastqData &data) {
  std::ofstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open output file: " + filepath);
  }

  // Split each field buffer by '\n' to recover individual record lines
  auto split = [](const std::vector<uint8_t> &buf) {
    std::vector<std::string> lines;
    std::string current;
    for (uint8_t c : buf) {
      if (c == '\n') {
        lines.push_back(std::move(current));
        current.clear();
      } else {
        current.push_back(static_cast<char>(c));
      }
    }
    if (!current.empty()) {
      lines.push_back(std::move(current));
    }
    return lines;
  };

  auto ids = split(data.identifiers);
  auto seqs = split(data.basecalls);
  auto quals = split(data.quality_scores);

  if (ids.size() != data.num_records || seqs.size() != data.num_records ||
      quals.size() != data.num_records) {
    throw std::runtime_error(
        "Record count mismatch during FASTQ reconstruction");
  }

  for (uint64_t i = 0; i < data.num_records; i++) {
    file << '@' << ids[i] << '\n'
         << seqs[i] << '\n'
         << '+' << '\n'
         << quals[i] << '\n';
  }
}

} // namespace gpufastq
