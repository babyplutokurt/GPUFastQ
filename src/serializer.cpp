#include "gpufastq/serializer.hpp"

#include <fstream>
#include <stdexcept>

namespace gpufastq {

namespace {

template <typename T> void write_val(std::ofstream &f, const T &val) {
  f.write(reinterpret_cast<const char *>(&val), sizeof(T));
}

template <typename T> T read_val(std::ifstream &f) {
  T val;
  f.read(reinterpret_cast<char *>(&val), sizeof(T));
  if (!f) {
    throw std::runtime_error("Unexpected end of file");
  }
  return val;
}

void write_blob(std::ofstream &f, const std::vector<uint8_t> &data) {
  if (!data.empty()) {
    f.write(reinterpret_cast<const char *>(data.data()), data.size());
  }
}

void write_index(std::ofstream &f, const std::vector<uint64_t> &index) {
  for (uint64_t value : index) {
    write_val(f, value);
  }
}

std::vector<uint8_t> read_blob(std::ifstream &f, size_t size) {
  std::vector<uint8_t> data(size);
  if (size > 0) {
    f.read(reinterpret_cast<char *>(data.data()), size);
    if (!f) {
      throw std::runtime_error("Unexpected end of file reading blob");
    }
  }
  return data;
}

std::vector<uint64_t> read_index(std::ifstream &f, size_t size) {
  std::vector<uint64_t> index(size);
  for (size_t i = 0; i < size; ++i) {
    index[i] = read_val<uint64_t>(f);
  }
  return index;
}

void validate_chunked_stream(const std::vector<uint8_t> &payload,
                             size_t original_size,
                             const std::vector<uint64_t> &chunk_sizes,
                             const char *name) {
  const bool has_payload = !payload.empty();
  const bool has_chunks = !chunk_sizes.empty();
  const bool has_original = original_size != 0;
  if (has_payload != has_chunks || has_payload != has_original) {
    throw std::runtime_error(std::string("Cannot serialize inconsistent ") +
                             name + " chunk metadata");
  }
}

} // namespace

void serialize(const std::string &filepath, const CompressedFastqData &data) {
  std::ofstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open output file: " + filepath);
  }

  validate_chunked_stream(data.identifiers.payload,
                          data.identifiers.original_size,
                          data.compressed_identifier_chunk_sizes,
                          "identifier");
  validate_chunked_stream(data.basecalls.payload,
                          data.basecalls.original_size,
                          data.compressed_basecall_chunk_sizes, "basecall");
  validate_chunked_stream(data.quality_scores.payload,
                          data.quality_scores.original_size,
                          data.compressed_quality_chunk_sizes, "quality");
  if ((!data.compressed_quality_chunk_sizes.empty()) !=
      (!data.uncompressed_quality_chunk_sizes.empty())) {
    throw std::runtime_error(
        "Cannot serialize inconsistent quality chunk size metadata");
  }
  if (data.compressed_quality_chunk_sizes.size() !=
      data.uncompressed_quality_chunk_sizes.size()) {
    throw std::runtime_error(
        "Cannot serialize mismatched quality chunk size vectors");
  }
  validate_chunked_stream(data.line_lengths.payload,
                          data.line_lengths.original_size,
                          data.compressed_line_length_chunk_sizes,
                          "line-length");

  write_val(file, MAGIC);
  write_val(file, FORMAT_VERSION);
  write_val(file, data.num_records);
  write_val(file, data.line_offset_count);

  write_val(file, static_cast<uint64_t>(data.identifiers.original_size));
  write_val(file, static_cast<uint64_t>(data.identifiers.payload.size()));
  write_val(file,
            static_cast<uint64_t>(data.compressed_identifier_chunk_sizes.size()));

  write_val(file, static_cast<uint64_t>(data.basecalls.original_size));
  write_val(file, static_cast<uint64_t>(data.basecalls.payload.size()));
  write_val(file,
            static_cast<uint64_t>(data.compressed_basecall_chunk_sizes.size()));

  write_val(file, static_cast<uint64_t>(data.quality_scores.original_size));
  write_val(file, static_cast<uint64_t>(data.quality_scores.payload.size()));
  write_val(file,
            static_cast<uint64_t>(data.compressed_quality_chunk_sizes.size()));

  write_val(file, static_cast<uint64_t>(data.line_lengths.original_size));
  write_val(file, static_cast<uint64_t>(data.line_lengths.payload.size()));
  write_val(file,
            static_cast<uint64_t>(data.compressed_line_length_chunk_sizes.size()));

  write_index(file, data.compressed_identifier_chunk_sizes);
  write_index(file, data.compressed_basecall_chunk_sizes);
  write_index(file, data.compressed_quality_chunk_sizes);
  write_index(file, data.uncompressed_quality_chunk_sizes);
  write_index(file, data.compressed_line_length_chunk_sizes);

  write_blob(file, data.identifiers.payload);
  write_blob(file, data.basecalls.payload);
  write_blob(file, data.quality_scores.payload);
  write_blob(file, data.line_lengths.payload);
}

CompressedFastqData deserialize(const std::string &filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open input file: " + filepath);
  }

  const uint32_t magic = read_val<uint32_t>(file);
  if (magic != MAGIC) {
    throw std::runtime_error("Invalid file format: bad magic number");
  }

  const uint32_t version = read_val<uint32_t>(file);
  if (version != FORMAT_VERSION) {
    throw std::runtime_error("Unsupported format version: " +
                             std::to_string(version));
  }

  CompressedFastqData data;
  data.num_records = read_val<uint64_t>(file);
  data.line_offset_count = read_val<uint64_t>(file);

  data.identifiers.original_size = read_val<uint64_t>(file);
  const uint64_t comp_id_size = read_val<uint64_t>(file);
  const uint64_t comp_id_chunks = read_val<uint64_t>(file);

  data.basecalls.original_size = read_val<uint64_t>(file);
  const uint64_t comp_seq_size = read_val<uint64_t>(file);
  const uint64_t comp_seq_chunks = read_val<uint64_t>(file);

  data.quality_scores.original_size = read_val<uint64_t>(file);
  const uint64_t comp_qual_size = read_val<uint64_t>(file);
  const uint64_t comp_qual_chunks = read_val<uint64_t>(file);

  data.line_lengths.original_size = read_val<uint64_t>(file);
  const uint64_t comp_index_size = read_val<uint64_t>(file);
  const uint64_t comp_index_chunks = read_val<uint64_t>(file);

  data.compressed_identifier_chunk_sizes = read_index(file, comp_id_chunks);
  data.compressed_basecall_chunk_sizes = read_index(file, comp_seq_chunks);
  data.compressed_quality_chunk_sizes = read_index(file, comp_qual_chunks);
  data.uncompressed_quality_chunk_sizes = read_index(file, comp_qual_chunks);
  data.compressed_line_length_chunk_sizes = read_index(file, comp_index_chunks);

  data.identifiers.payload = read_blob(file, comp_id_size);
  data.basecalls.payload = read_blob(file, comp_seq_size);
  data.quality_scores.payload = read_blob(file, comp_qual_size);
  data.line_lengths.payload = read_blob(file, comp_index_size);

  return data;
}

} // namespace gpufastq
