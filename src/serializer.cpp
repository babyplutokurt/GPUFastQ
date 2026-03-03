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
                             const std::vector<uint64_t> &chunk_sizes,
                             const char *name) {
  const bool has_payload = !payload.empty();
  const bool has_chunks = !chunk_sizes.empty();
  if (has_payload != has_chunks) {
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

  validate_chunked_stream(data.compressed_identifiers,
                          data.compressed_identifier_chunk_sizes,
                          "identifier");
  validate_chunked_stream(data.compressed_basecalls,
                          data.compressed_basecall_chunk_sizes, "basecall");
  validate_chunked_stream(data.compressed_quality,
                          data.compressed_quality_chunk_sizes, "quality");
  validate_chunked_stream(data.compressed_line_offsets,
                          data.compressed_line_offset_chunk_sizes,
                          "line-offset");

  write_val(file, MAGIC);
  write_val(file, FORMAT_VERSION);
  write_val(file, data.num_records);
  write_val(file, data.line_offset_count);

  write_val(file, data.original_id_size);
  write_val(file, static_cast<uint64_t>(data.compressed_identifiers.size()));
  write_val(file,
            static_cast<uint64_t>(data.compressed_identifier_chunk_sizes.size()));

  write_val(file, data.original_seq_size);
  write_val(file, static_cast<uint64_t>(data.compressed_basecalls.size()));
  write_val(file,
            static_cast<uint64_t>(data.compressed_basecall_chunk_sizes.size()));

  write_val(file, data.original_qual_size);
  write_val(file, static_cast<uint64_t>(data.compressed_quality.size()));
  write_val(file,
            static_cast<uint64_t>(data.compressed_quality_chunk_sizes.size()));

  write_val(file, data.original_index_size);
  write_val(file, static_cast<uint64_t>(data.compressed_line_offsets.size()));
  write_val(file,
            static_cast<uint64_t>(data.compressed_line_offset_chunk_sizes.size()));

  write_index(file, data.compressed_identifier_chunk_sizes);
  write_index(file, data.compressed_basecall_chunk_sizes);
  write_index(file, data.compressed_quality_chunk_sizes);
  write_index(file, data.compressed_line_offset_chunk_sizes);

  write_blob(file, data.compressed_identifiers);
  write_blob(file, data.compressed_basecalls);
  write_blob(file, data.compressed_quality);
  write_blob(file, data.compressed_line_offsets);
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

  data.original_id_size = read_val<uint64_t>(file);
  const uint64_t comp_id_size = read_val<uint64_t>(file);
  const uint64_t comp_id_chunks = read_val<uint64_t>(file);

  data.original_seq_size = read_val<uint64_t>(file);
  const uint64_t comp_seq_size = read_val<uint64_t>(file);
  const uint64_t comp_seq_chunks = read_val<uint64_t>(file);

  data.original_qual_size = read_val<uint64_t>(file);
  const uint64_t comp_qual_size = read_val<uint64_t>(file);
  const uint64_t comp_qual_chunks = read_val<uint64_t>(file);

  data.original_index_size = read_val<uint64_t>(file);
  const uint64_t comp_index_size = read_val<uint64_t>(file);
  const uint64_t comp_index_chunks = read_val<uint64_t>(file);

  data.compressed_identifier_chunk_sizes = read_index(file, comp_id_chunks);
  data.compressed_basecall_chunk_sizes = read_index(file, comp_seq_chunks);
  data.compressed_quality_chunk_sizes = read_index(file, comp_qual_chunks);
  data.compressed_line_offset_chunk_sizes = read_index(file, comp_index_chunks);

  data.compressed_identifiers = read_blob(file, comp_id_size);
  data.compressed_basecalls = read_blob(file, comp_seq_size);
  data.compressed_quality = read_blob(file, comp_qual_size);
  data.compressed_line_offsets = read_blob(file, comp_index_size);

  return data;
}

} // namespace gpufastq
