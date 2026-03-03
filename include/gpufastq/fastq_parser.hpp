#pragma once

#include "fastq_record.hpp"
#include <string>

namespace gpufastq {

/// Parse a FASTQ file, splitting records into identifier/basecall/quality
/// vectors
FastqData parse_fastq(const std::string &filepath);

/// Reconstruct a FASTQ file from separated field vectors
void write_fastq(const std::string &filepath, const FastqData &data);

} // namespace gpufastq
