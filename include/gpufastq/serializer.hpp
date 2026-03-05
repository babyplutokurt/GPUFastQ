#pragma once

#include "fastq_record.hpp"
#include <cstdint>
#include <string>

namespace gpufastq {

/// File extension for compressed FASTQ
constexpr const char *COMPRESSED_EXTENSION = ".gpufq";

/// Magic bytes: "GFQZ" in little-endian
constexpr uint32_t MAGIC = 0x5A514647;

/// File format version
constexpr uint32_t FORMAT_VERSION = 15;

/// Serialize compressed FASTQ data to a binary .gpufq file
void serialize(const std::string &filepath, const CompressedFastqData &data);

/// Deserialize compressed FASTQ data from a .gpufq file
CompressedFastqData deserialize(const std::string &filepath);

} // namespace gpufastq
