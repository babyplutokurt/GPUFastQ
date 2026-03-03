#include "gpufastq/codec_bsc.hpp"
#include "gpufastq/codec_gpu.cuh"
#include "gpufastq/codec_gpu_nvcomp.cuh"
#include "gpufastq/fastq_parser.hpp"
#include "gpufastq/gpu_compressor.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace gpufastq {

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +      \
                               ":" + std::to_string(__LINE__) + ": " +         \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

namespace {

constexpr size_t MAX_FIELD_SLICE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t BASECALL_N_BLOCK_SIZE = 8192;

struct ChunkedCompressedBuffer {
  std::vector<uint8_t> data;
  std::vector<uint64_t> chunk_sizes;
};

struct TokenizedIdentifier {
  std::vector<std::string> tokens;
  std::vector<std::string> separators;
};

struct IdentifierColumnBuffers {
  IdentifierColumnKind kind = IdentifierColumnKind::String;
  std::vector<uint8_t> string_values;
  std::vector<uint32_t> string_lengths;
  std::vector<int32_t> int_values;
};

std::vector<uint8_t>
gpu_decompress_chunked(const std::vector<uint8_t> &compressed,
                       const std::vector<uint64_t> &chunk_sizes,
                       uint64_t expected_size);

struct DeviceFieldBuffers {
  uint8_t *identifiers = nullptr;
  uint8_t *basecalls = nullptr;
  uint8_t *quality_scores = nullptr;
};

struct EncodedBasecallBuffers {
  uint8_t *packed_bases = nullptr;
  uint32_t *n_counts = nullptr;
  uint64_t *n_offsets = nullptr;
  uint16_t *n_positions = nullptr;
};

uint64_t sum_sizes(const std::vector<uint64_t> &sizes) {
  return std::accumulate(sizes.begin(), sizes.end(), uint64_t{0});
}

const char *identifier_column_kind_name(IdentifierColumnKind kind) {
  switch (kind) {
  case IdentifierColumnKind::String:
    return "string";
  case IdentifierColumnKind::Int32:
    return "int32";
  }
  return "unknown";
}

const char *identifier_column_encoding_name(IdentifierColumnEncoding encoding) {
  switch (encoding) {
  case IdentifierColumnEncoding::Plain:
    return "plain";
  case IdentifierColumnEncoding::Delta:
    return "delta";
  }
  return "unknown";
}

double compression_ratio_percent(uint64_t compressed_size, uint64_t raw_size) {
  if (raw_size == 0) {
    return 0.0;
  }
  return 100.0 * static_cast<double>(compressed_size) /
         static_cast<double>(raw_size);
}

uint64_t line_content_length(const std::vector<uint64_t> &line_offsets,
                             uint64_t line_idx) {
  return line_offsets[line_idx + 1] - line_offsets[line_idx] - 1;
}

bool is_identifier_separator(uint8_t ch) {
  return ch == ':' || ch == '/' || ch == '-' || ch == '.' ||
         std::isspace(static_cast<unsigned char>(ch)) != 0;
}

TokenizedIdentifier tokenize_identifier(const uint8_t *data, size_t size) {
  TokenizedIdentifier out;
  std::string token;
  std::string separator;
  bool in_separator = false;

  for (size_t i = 0; i < size; ++i) {
    const char ch = static_cast<char>(data[i]);
    if (is_identifier_separator(data[i])) {
      if (!token.empty()) {
        out.tokens.push_back(std::move(token));
        token.clear();
      }
      separator.push_back(ch);
      in_separator = true;
      continue;
    }

    if (in_separator) {
      out.separators.push_back(std::move(separator));
      separator.clear();
      in_separator = false;
    }
    token.push_back(ch);
  }

  if (!token.empty()) {
    out.tokens.push_back(std::move(token));
  }
  if (!separator.empty()) {
    out.separators.push_back(std::move(separator));
  }

  return out;
}

bool token_is_int32(const std::string &token) {
  if (token.empty()) {
    return false;
  }

  size_t index = 0;
  if (token[0] == '+' || token[0] == '-') {
    if (token.size() == 1) {
      return false;
    }
    index = 1;
  }
  for (; index < token.size(); ++index) {
    if (!std::isdigit(static_cast<unsigned char>(token[index]))) {
      return false;
    }
  }
  try {
    const long long value = std::stoll(token);
    return value >= std::numeric_limits<int32_t>::min() &&
           value <= std::numeric_limits<int32_t>::max() &&
           std::to_string(value) == token;
  } catch (...) {
    return false;
  }
}

int32_t parse_int32_token(const std::string &token) {
  const long long value = std::stoll(token);
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max() ||
      std::to_string(value) != token) {
    throw std::runtime_error("Identifier token is not a canonical int32: " +
                             token);
  }
  return static_cast<int32_t>(value);
}

std::vector<uint8_t> int32_vector_to_bytes(const std::vector<int32_t> &values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(int32_t));
  if (!values.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
}

std::vector<int32_t> delta_encode_int32(const std::vector<int32_t> &values) {
  std::vector<int32_t> deltas(values.size());
  if (values.empty()) {
    return deltas;
  }

  deltas[0] = values[0];
  for (size_t i = 1; i < values.size(); ++i) {
    const int64_t delta = static_cast<int64_t>(values[i]) -
                          static_cast<int64_t>(values[i - 1]);
    if (delta < std::numeric_limits<int32_t>::min() ||
        delta > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error("Identifier delta exceeds int32 range");
    }
    deltas[i] = static_cast<int32_t>(delta);
  }
  return deltas;
}

std::vector<int32_t> delta_decode_int32(const int32_t *values, size_t count) {
  std::vector<int32_t> decoded(count);
  if (count == 0) {
    return decoded;
  }

  int64_t running = values[0];
  if (running < std::numeric_limits<int32_t>::min() ||
      running > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error("Identifier delta decode overflow");
  }
  decoded[0] = static_cast<int32_t>(running);
  for (size_t i = 1; i < count; ++i) {
    running += values[i];
    if (running < std::numeric_limits<int32_t>::min() ||
        running > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error("Identifier delta decode overflow");
    }
    decoded[i] = static_cast<int32_t>(running);
  }
  return decoded;
}

template <typename T> void cuda_free_if_set(T *ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

ChunkedCompressedBuffer host_compress_chunked(const std::vector<uint8_t> &input,
                                              size_t field_slice_size,
                                              size_t nvcomp_chunk_size) {
  ChunkedCompressedBuffer result;
  if (input.empty()) {
    return result;
  }
  if (field_slice_size == 0 || field_slice_size > MAX_FIELD_SLICE_SIZE) {
    throw std::runtime_error(
        "Requested field slice size is out of supported range");
  }

  for (size_t offset = 0; offset < input.size(); offset += field_slice_size) {
    const size_t slice_size = std::min(field_slice_size, input.size() - offset);
    std::vector<uint8_t> slice(
        input.begin() + static_cast<std::ptrdiff_t>(offset),
        input.begin() + static_cast<std::ptrdiff_t>(offset + slice_size));
    auto compressed = nvcomp_zstd_compress(slice, nvcomp_chunk_size);
    result.chunk_sizes.push_back(compressed.payload.size());
    result.data.insert(result.data.end(), compressed.payload.begin(),
                       compressed.payload.end());
  }

  return result;
}

std::vector<uint8_t> extract_flat_identifiers_host(const FastqData &data) {
  const FastqFieldStats stats = compute_field_stats(data);
  std::vector<uint8_t> identifiers(static_cast<size_t>(stats.identifiers_size));

  uint64_t offset = 0;
  for (uint64_t record = 0; record < data.num_records; ++record) {
    const uint64_t id_line = 4 * record;
    const uint64_t id_len = line_content_length(data.line_offsets, id_line) - 1;
    const uint64_t id_start = data.line_offsets[id_line] + 1;
    std::memcpy(identifiers.data() + offset, data.raw_bytes.data() + id_start,
                static_cast<size_t>(id_len));
    offset += id_len;
  }

  return identifiers;
}

CompressedIdentifierData compress_identifiers_flat(const FastqData &data,
                                                   size_t field_slice_size,
                                                   size_t nvcomp_chunk_size) {
  CompressedIdentifierData result;
  result.mode = IdentifierCompressionMode::Flat;
  result.original_size = compute_field_stats(data).identifiers_size;
  result.flat_data.original_size = result.original_size;
  auto identifiers = extract_flat_identifiers_host(data);
  auto chunks =
      host_compress_chunked(identifiers, field_slice_size, nvcomp_chunk_size);
  result.flat_data.payload = std::move(chunks.data);
  result.compressed_flat_chunk_sizes = std::move(chunks.chunk_sizes);
  return result;
}

CompressedIdentifierData compress_identifiers_columnar(
    const FastqData &data, size_t field_slice_size, size_t nvcomp_chunk_size) {
  if (!data.identifier_layout.columnar ||
      data.identifier_layout.column_kinds.empty()) {
    return compress_identifiers_flat(data, field_slice_size, nvcomp_chunk_size);
  }

  std::vector<IdentifierColumnBuffers> columns(
      data.identifier_layout.column_kinds.size());
  for (size_t i = 0; i < columns.size(); ++i) {
    columns[i].kind = data.identifier_layout.column_kinds[i];
    if (columns[i].kind == IdentifierColumnKind::String) {
      columns[i].string_lengths.reserve(static_cast<size_t>(data.num_records));
    } else {
      columns[i].int_values.reserve(static_cast<size_t>(data.num_records));
    }
  }

  for (uint64_t record = 0; record < data.num_records; ++record) {
    const uint64_t id_line = 4 * record;
    const uint64_t id_len = line_content_length(data.line_offsets, id_line);
    const uint64_t id_start = data.line_offsets[id_line] + 1;
    auto tokenized = tokenize_identifier(data.raw_bytes.data() + id_start,
                                         static_cast<size_t>(id_len - 1));
    if (tokenized.tokens.size() != data.identifier_layout.column_kinds.size() ||
        tokenized.separators != data.identifier_layout.separators) {
      return compress_identifiers_flat(data, field_slice_size, nvcomp_chunk_size);
    }

    for (size_t i = 0; i < columns.size(); ++i) {
      if (columns[i].kind == IdentifierColumnKind::String) {
        const auto &token = tokenized.tokens[i];
        if (token.size() > std::numeric_limits<uint32_t>::max()) {
          throw std::runtime_error("Identifier token length exceeds uint32_t");
        }
        columns[i].string_lengths.push_back(static_cast<uint32_t>(token.size()));
        columns[i].string_values.insert(columns[i].string_values.end(),
                                        token.begin(), token.end());
      } else {
        if (!token_is_int32(tokenized.tokens[i])) {
          return compress_identifiers_flat(data, field_slice_size, nvcomp_chunk_size);
        }
        columns[i].int_values.push_back(parse_int32_token(tokenized.tokens[i]));
      }
    }
  }

  CompressedIdentifierData result;
  result.mode = IdentifierCompressionMode::Columnar;
  result.original_size = compute_field_stats(data).identifiers_size;
  result.layout = data.identifier_layout;
  result.layout.columnar = true;
  result.columns.resize(columns.size());

  for (size_t i = 0; i < columns.size(); ++i) {
    auto &out = result.columns[i];
    out.kind = columns[i].kind;
    if (out.kind == IdentifierColumnKind::String) {
      out.encoding = IdentifierColumnEncoding::Plain;
      out.values.original_size = columns[i].string_values.size();
      auto value_chunks = host_compress_chunked(columns[i].string_values,
                                                field_slice_size,
                                                nvcomp_chunk_size);
      out.values.payload = std::move(value_chunks.data);
      out.compressed_value_chunk_sizes = std::move(value_chunks.chunk_sizes);

      std::vector<uint8_t> length_bytes(columns[i].string_lengths.size() *
                                        sizeof(uint32_t));
      if (!columns[i].string_lengths.empty()) {
        std::memcpy(length_bytes.data(), columns[i].string_lengths.data(),
                    length_bytes.size());
      }
      out.lengths.original_size = length_bytes.size();
      auto length_chunks =
          host_compress_chunked(length_bytes, field_slice_size, nvcomp_chunk_size);
      out.lengths.payload = std::move(length_chunks.data);
      out.compressed_length_chunk_sizes = std::move(length_chunks.chunk_sizes);
    } else {
      const auto plain_bytes = int32_vector_to_bytes(columns[i].int_values);
      const auto delta_values = delta_encode_int32(columns[i].int_values);
      const auto delta_bytes = int32_vector_to_bytes(delta_values);

      auto plain_chunks =
          host_compress_chunked(plain_bytes, field_slice_size, nvcomp_chunk_size);
      auto delta_chunks =
          host_compress_chunked(delta_bytes, field_slice_size, nvcomp_chunk_size);

      out.values.original_size = plain_bytes.size();
      if (sum_sizes(delta_chunks.chunk_sizes) < sum_sizes(plain_chunks.chunk_sizes)) {
        out.encoding = IdentifierColumnEncoding::Delta;
        out.values.payload = std::move(delta_chunks.data);
        out.compressed_value_chunk_sizes = std::move(delta_chunks.chunk_sizes);
      } else {
        out.encoding = IdentifierColumnEncoding::Plain;
        out.values.payload = std::move(plain_chunks.data);
        out.compressed_value_chunk_sizes = std::move(plain_chunks.chunk_sizes);
      }
    }
  }

  return result;
}

std::vector<uint8_t> decompress_identifiers(const CompressedIdentifierData &data,
                                            uint64_t num_records) {
  if (data.mode == IdentifierCompressionMode::Flat) {
    return gpu_decompress_chunked(data.flat_data.payload,
                                  data.compressed_flat_chunk_sizes,
                                  data.flat_data.original_size);
  }
  if (data.mode != IdentifierCompressionMode::Columnar) {
    throw std::runtime_error("Unknown identifier compression mode");
  }
  if (data.layout.column_kinds.empty() || data.columns.empty() ||
      data.layout.column_kinds.size() != data.columns.size() ||
      data.layout.separators.size() + 1 != data.columns.size()) {
    throw std::runtime_error("Decoded identifier column metadata is invalid");
  }

  struct DecodedColumn {
    IdentifierColumnKind kind = IdentifierColumnKind::String;
    IdentifierColumnEncoding encoding = IdentifierColumnEncoding::Plain;
    std::vector<uint8_t> value_bytes;
    std::vector<uint32_t> lengths;
    std::vector<int32_t> decoded_int_values;
    const int32_t *int_values = nullptr;
    size_t value_offset = 0;
    size_t record_count = 0;
  };

  std::vector<DecodedColumn> columns(data.columns.size());
  for (size_t i = 0; i < data.columns.size(); ++i) {
    columns[i].kind = data.columns[i].kind;
    columns[i].encoding = data.columns[i].encoding;
    columns[i].value_bytes = gpu_decompress_chunked(
        data.columns[i].values.payload, data.columns[i].compressed_value_chunk_sizes,
        data.columns[i].values.original_size);
    if (columns[i].kind == IdentifierColumnKind::String) {
      const auto length_bytes = gpu_decompress_chunked(
          data.columns[i].lengths.payload,
          data.columns[i].compressed_length_chunk_sizes,
          data.columns[i].lengths.original_size);
      if (length_bytes.size() != num_records * sizeof(uint32_t)) {
        throw std::runtime_error(
            "Decoded identifier string-length payload has an unexpected size");
      }
      columns[i].lengths.resize(static_cast<size_t>(num_records));
      if (!columns[i].lengths.empty()) {
        std::memcpy(columns[i].lengths.data(), length_bytes.data(),
                    length_bytes.size());
      }
    } else {
      if (columns[i].value_bytes.size() != num_records * sizeof(int32_t)) {
        throw std::runtime_error(
            "Decoded identifier numeric payload has an unexpected size");
      }
      const auto *encoded_values =
          reinterpret_cast<const int32_t *>(columns[i].value_bytes.data());
      if (columns[i].encoding == IdentifierColumnEncoding::Delta) {
        columns[i].decoded_int_values =
            delta_decode_int32(encoded_values, static_cast<size_t>(num_records));
        columns[i].int_values = columns[i].decoded_int_values.data();
      } else {
        columns[i].int_values = encoded_values;
      }
    }
  }

  std::vector<uint8_t> identifiers;
  identifiers.reserve(static_cast<size_t>(data.original_size));
  for (uint64_t record = 0; record < num_records; ++record) {
    for (size_t column = 0; column < columns.size(); ++column) {
      if (columns[column].kind == IdentifierColumnKind::String) {
        const uint32_t len = columns[column].lengths[static_cast<size_t>(record)];
        if (columns[column].value_offset + len > columns[column].value_bytes.size()) {
          throw std::runtime_error(
              "Decoded identifier string column exceeds its payload");
        }
        identifiers.insert(
            identifiers.end(),
            columns[column].value_bytes.begin() +
                static_cast<std::ptrdiff_t>(columns[column].value_offset),
            columns[column].value_bytes.begin() +
                static_cast<std::ptrdiff_t>(columns[column].value_offset + len));
        columns[column].value_offset += len;
      } else {
        const auto token = std::to_string(columns[column].int_values[record]);
        identifiers.insert(identifiers.end(), token.begin(), token.end());
      }

      if (column + 1 < data.layout.separators.size() + 1) {
        const auto &sep = data.layout.separators[column];
        identifiers.insert(identifiers.end(), sep.begin(), sep.end());
      }
    }
  }

  if (identifiers.size() != data.original_size) {
    throw std::runtime_error(
        "Decoded identifier columns reconstructed an unexpected size");
  }
  return identifiers;
}

__global__ void compute_field_lengths_kernel(const uint64_t *line_offsets,
                                             uint64_t *identifier_lengths,
                                             uint64_t *basecall_lengths,
                                             uint64_t *quality_lengths,
                                             uint64_t num_records) {
  const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_records) {
    return;
  }

  const uint64_t id_line = 4 * idx;
  const uint64_t seq_line = id_line + 1;
  const uint64_t plus_line = id_line + 2;
  const uint64_t qual_line = id_line + 3;

  identifier_lengths[idx] = line_offsets[seq_line] - line_offsets[id_line] - 2;
  basecall_lengths[idx] = line_offsets[plus_line] - line_offsets[seq_line] - 1;
  quality_lengths[idx] =
      line_offsets[qual_line + 1] - line_offsets[qual_line] - 1;
}

__global__ void gather_fields_kernel(
    const uint8_t *raw_bytes, const uint64_t *line_offsets,
    const uint64_t *identifier_offsets, const uint64_t *basecall_offsets,
    const uint64_t *quality_offsets, const uint64_t *identifier_lengths,
    const uint64_t *basecall_lengths, const uint64_t *quality_lengths,
    uint8_t *identifiers, uint8_t *basecalls, uint8_t *quality_scores,
    uint64_t num_records) {
  const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_records) {
    return;
  }

  const uint64_t id_line = 4 * idx;
  const uint64_t seq_line = id_line + 1;
  const uint64_t qual_line = id_line + 3;

  const uint64_t id_src = line_offsets[id_line] + 1;
  const uint64_t seq_src = line_offsets[seq_line];
  const uint64_t qual_src = line_offsets[qual_line];

  const uint64_t id_dst = identifier_offsets[idx];
  const uint64_t seq_dst = basecall_offsets[idx];
  const uint64_t qual_dst = quality_offsets[idx];

  if (identifiers != nullptr) {
    for (uint64_t i = 0; i < identifier_lengths[idx]; ++i) {
      identifiers[id_dst + i] = raw_bytes[id_src + i];
    }
  }
  for (uint64_t i = 0; i < basecall_lengths[idx]; ++i) {
    basecalls[seq_dst + i] = raw_bytes[seq_src + i];
  }
  for (uint64_t i = 0; i < quality_lengths[idx]; ++i) {
    quality_scores[qual_dst + i] = raw_bytes[qual_src + i];
  }
}

__device__ inline uint8_t encode_basecall_2bit(uint8_t base,
                                               bool *is_n,
                                               bool *is_valid) {
  switch (base) {
  case 'A':
  case 'a':
    *is_n = false;
    *is_valid = true;
    return 0;
  case 'C':
  case 'c':
    *is_n = false;
    *is_valid = true;
    return 1;
  case 'G':
  case 'g':
    *is_n = false;
    *is_valid = true;
    return 2;
  case 'T':
  case 't':
    *is_n = false;
    *is_valid = true;
    return 3;
  case 'N':
  case 'n':
    *is_n = true;
    *is_valid = true;
    return 0;
  default:
    *is_n = false;
    *is_valid = false;
    return 0;
  }
}

__global__ void count_n_basecalls_kernel(const uint8_t *basecalls,
                                         uint64_t basecall_count,
                                         uint32_t *n_counts,
                                         uint64_t *invalid_position) {
  const uint64_t block_index = blockIdx.x;
  const uint64_t block_start =
      block_index * static_cast<uint64_t>(BASECALL_N_BLOCK_SIZE);
  if (block_start >= basecall_count) {
    return;
  }

  __shared__ uint32_t shared_count;
  if (threadIdx.x == 0) {
    shared_count = 0;
  }
  __syncthreads();

  const uint64_t block_end = min(
      block_start + static_cast<uint64_t>(BASECALL_N_BLOCK_SIZE), basecall_count);
  uint32_t local_count = 0;
  for (uint64_t index = block_start + threadIdx.x; index < block_end;
       index += blockDim.x) {
    bool is_n = false;
    bool is_valid = false;
    encode_basecall_2bit(basecalls[index], &is_n, &is_valid);
    if (!is_valid) {
      atomicMin(reinterpret_cast<unsigned long long *>(invalid_position),
                static_cast<unsigned long long>(index));
      continue;
    }
    if (is_n) {
      ++local_count;
    }
  }

  if (local_count != 0) {
    atomicAdd(&shared_count, local_count);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    n_counts[block_index] = shared_count;
  }
}

__global__ void pack_basecalls_2bit_kernel(const uint8_t *basecalls,
                                           uint64_t basecall_count,
                                           uint8_t *packed_bases,
                                           uint64_t *invalid_position) {
  const uint64_t packed_index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t base_index = packed_index * 4;
  if (base_index >= basecall_count) {
    return;
  }

  uint8_t packed_value = 0;
  for (uint32_t lane = 0; lane < 4; ++lane) {
    const uint64_t index = base_index + lane;
    if (index >= basecall_count) {
      break;
    }

    bool is_n = false;
    bool is_valid = false;
    const uint8_t code =
        encode_basecall_2bit(basecalls[index], &is_n, &is_valid);
    if (!is_valid) {
      atomicMin(reinterpret_cast<unsigned long long *>(invalid_position),
                static_cast<unsigned long long>(index));
      return;
    }
    packed_value |= static_cast<uint8_t>(code << (2 * lane));
  }
  packed_bases[packed_index] = packed_value;
}

__global__ void scatter_n_positions_kernel(const uint8_t *basecalls,
                                           uint64_t basecall_count,
                                           const uint64_t *n_offsets,
                                           uint16_t *n_positions,
                                           uint64_t *invalid_position) {
  const uint64_t block_index = blockIdx.x;
  const uint64_t block_start =
      block_index * static_cast<uint64_t>(BASECALL_N_BLOCK_SIZE);
  if (block_start >= basecall_count) {
    return;
  }

  __shared__ uint32_t cursor;
  if (threadIdx.x == 0) {
    cursor = 0;
  }
  __syncthreads();

  const uint64_t block_end = min(
      block_start + static_cast<uint64_t>(BASECALL_N_BLOCK_SIZE), basecall_count);
  for (uint64_t index = block_start + threadIdx.x; index < block_end;
       index += blockDim.x) {
    bool is_n = false;
    bool is_valid = false;
    encode_basecall_2bit(basecalls[index], &is_n, &is_valid);
    if (!is_valid) {
      atomicMin(reinterpret_cast<unsigned long long *>(invalid_position),
                static_cast<unsigned long long>(index));
      continue;
    }
    if (is_n) {
      const uint32_t slot = atomicAdd(&cursor, 1u);
      n_positions[n_offsets[block_index] + slot] =
          static_cast<uint16_t>(index - block_start);
    }
  }
}

ChunkedCompressedBuffer gpu_compress_device_chunked(const uint8_t *d_input,
                                                    size_t input_size,
                                                    size_t field_slice_size,
                                                    size_t nvcomp_chunk_size,
                                                    cudaStream_t stream) {
  ChunkedCompressedBuffer result;
  if (input_size == 0) {
    return result;
  }
  if (field_slice_size == 0 || field_slice_size > MAX_FIELD_SLICE_SIZE) {
    throw std::runtime_error(
        "Requested field slice size is out of supported range");
  }

  for (size_t offset = 0; offset < input_size; offset += field_slice_size) {
    const size_t slice_size = std::min(field_slice_size, input_size - offset);
    auto compressed = nvcomp_zstd_compress_device(d_input + offset, slice_size,
                                                  nvcomp_chunk_size, stream);
    result.chunk_sizes.push_back(compressed.payload.size());
    result.data.insert(result.data.end(), compressed.payload.begin(),
                       compressed.payload.end());
  }

  return result;
}

std::vector<uint8_t>
gpu_decompress_chunked(const std::vector<uint8_t> &compressed,
                       const std::vector<uint64_t> &chunk_sizes,
                       uint64_t expected_size) {
  if (compressed.empty()) {
    if (!chunk_sizes.empty() || expected_size != 0) {
      throw std::runtime_error(
          "Compressed chunk metadata is inconsistent for empty payload");
    }
    return {};
  }

  if (sum_sizes(chunk_sizes) != compressed.size()) {
    throw std::runtime_error(
        "Compressed chunk sizes do not match payload size");
  }

  std::vector<uint8_t> output;
  output.reserve(expected_size);

  size_t offset = 0;
  for (uint64_t chunk_size : chunk_sizes) {
    std::vector<uint8_t> slice(
        compressed.begin() + static_cast<std::ptrdiff_t>(offset),
        compressed.begin() + static_cast<std::ptrdiff_t>(offset + chunk_size));
    const size_t expected_chunk_size = static_cast<size_t>(std::min<uint64_t>(
        MAX_FIELD_SLICE_SIZE, expected_size - output.size()));
    ZstdCompressedBlock block{std::move(slice), expected_chunk_size};
    auto decompressed = nvcomp_zstd_decompress(block);
    output.insert(output.end(), decompressed.begin(), decompressed.end());
    offset += chunk_size;
  }

  if (offset != compressed.size() || output.size() != expected_size) {
    throw std::runtime_error(
        "Chunked decompression produced an unexpected size");
  }

  return output;
}

CompressedBasecallData compress_basecalls_device(const uint8_t *d_basecalls,
                                                 uint64_t basecall_count,
                                                 size_t field_slice_size,
                                                 size_t nvcomp_chunk_size,
                                                 cudaStream_t stream) {
  CompressedBasecallData result;
  result.original_size = basecall_count;
  result.n_block_size = BASECALL_N_BLOCK_SIZE;
  if (basecall_count == 0) {
    return result;
  }

  const uint64_t packed_size = (basecall_count + 3) / 4;
  const uint64_t block_count =
      (basecall_count + BASECALL_N_BLOCK_SIZE - 1) / BASECALL_N_BLOCK_SIZE;
  result.packed_bases.original_size = packed_size;

  EncodedBasecallBuffers encoded;
  uint64_t *d_invalid_position = nullptr;
  uint64_t invalid_position = std::numeric_limits<uint64_t>::max();

  try {
    CUDA_CHECK(cudaMalloc(&encoded.packed_bases, packed_size));
    CUDA_CHECK(cudaMalloc(&encoded.n_counts, block_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&encoded.n_offsets, block_count * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_invalid_position, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_invalid_position, &invalid_position,
                               sizeof(uint64_t), cudaMemcpyHostToDevice,
                               stream));

    const uint32_t kernel_block_size = 256;
    count_n_basecalls_kernel<<<static_cast<uint32_t>(block_count),
                               kernel_block_size, 0, stream>>>(
        d_basecalls, basecall_count, encoded.n_counts, d_invalid_position);
    CUDA_CHECK(cudaGetLastError());

    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                           thrust::device_pointer_cast(encoded.n_counts),
                           thrust::device_pointer_cast(encoded.n_counts) +
                               block_count,
                           thrust::device_pointer_cast(encoded.n_offsets));

    std::vector<uint32_t> h_n_counts(block_count);
    CUDA_CHECK(cudaMemcpyAsync(h_n_counts.data(), encoded.n_counts,
                               block_count * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));

    const uint32_t pack_grid_size = static_cast<uint32_t>(
        (packed_size + kernel_block_size - 1) / kernel_block_size);
    pack_basecalls_2bit_kernel<<<pack_grid_size, kernel_block_size, 0, stream>>>(
        d_basecalls, basecall_count, encoded.packed_bases, d_invalid_position);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&invalid_position, d_invalid_position,
                          sizeof(uint64_t), cudaMemcpyDeviceToHost));
    if (invalid_position != std::numeric_limits<uint64_t>::max()) {
      throw std::runtime_error("Encountered a non-ACGTN basecall at offset " +
                               std::to_string(invalid_position));
    }

    uint64_t total_n_count = 0;
    std::vector<uint16_t> h_n_counts_16(block_count);
    for (uint64_t i = 0; i < block_count; ++i) {
      if (h_n_counts[i] > BASECALL_N_BLOCK_SIZE) {
        throw std::runtime_error("Basecall N-count overflowed its block size");
      }
      h_n_counts_16[i] = static_cast<uint16_t>(h_n_counts[i]);
      total_n_count += h_n_counts[i];
    }

    std::vector<uint8_t> h_n_count_bytes(block_count * sizeof(uint16_t));
    if (!h_n_counts_16.empty()) {
      std::memcpy(h_n_count_bytes.data(), h_n_counts_16.data(),
                  h_n_count_bytes.size());
    }
    result.n_counts =
        nvcomp_zstd_compress(h_n_count_bytes, nvcomp_chunk_size, stream);

    if (total_n_count > 0) {
      result.n_positions.original_size = total_n_count * sizeof(uint16_t);
      CUDA_CHECK(
          cudaMalloc(&encoded.n_positions, result.n_positions.original_size));
      scatter_n_positions_kernel<<<static_cast<uint32_t>(block_count),
                                   kernel_block_size, 0, stream>>>(
          d_basecalls, basecall_count, encoded.n_offsets, encoded.n_positions,
          d_invalid_position);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaMemcpy(&invalid_position, d_invalid_position,
                            sizeof(uint64_t), cudaMemcpyDeviceToHost));
      if (invalid_position != std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error("Encountered a non-ACGTN basecall at offset " +
                                 std::to_string(invalid_position));
      }
    }

    auto packed_chunks = gpu_compress_device_chunked(
        encoded.packed_bases, packed_size, field_slice_size, nvcomp_chunk_size,
        stream);
    result.packed_bases.payload = std::move(packed_chunks.data);
    result.compressed_packed_chunk_sizes = std::move(packed_chunks.chunk_sizes);

    if (result.n_positions.original_size > 0) {
      auto n_position_chunks = gpu_compress_device_chunked(
          reinterpret_cast<const uint8_t *>(encoded.n_positions),
          result.n_positions.original_size, field_slice_size, nvcomp_chunk_size,
          stream);
      result.n_positions.payload = std::move(n_position_chunks.data);
      result.compressed_n_position_chunk_sizes =
          std::move(n_position_chunks.chunk_sizes);
    }
  } catch (...) {
    cuda_free_if_set(d_invalid_position);
    cuda_free_if_set(encoded.n_positions);
    cuda_free_if_set(encoded.n_offsets);
    cuda_free_if_set(encoded.n_counts);
    cuda_free_if_set(encoded.packed_bases);
    throw;
  }

  cuda_free_if_set(d_invalid_position);
  cuda_free_if_set(encoded.n_positions);
  cuda_free_if_set(encoded.n_offsets);
  cuda_free_if_set(encoded.n_counts);
  cuda_free_if_set(encoded.packed_bases);
  return result;
}

std::vector<uint8_t>
decode_basecalls(const CompressedBasecallData &compressed,
                 const std::vector<uint8_t> &n_count_bytes,
                 const std::vector<uint8_t> &packed_bases,
                 const std::vector<uint8_t> &n_position_bytes) {
  if (compressed.original_size == 0) {
    return {};
  }

  if (compressed.n_block_size == 0) {
    throw std::runtime_error("Decoded basecall metadata is missing block size");
  }

  const uint64_t expected_block_count =
      (compressed.original_size + compressed.n_block_size - 1) /
      compressed.n_block_size;
  if (n_count_bytes.size() != expected_block_count * sizeof(uint16_t)) {
    throw std::runtime_error("Decoded basecall N-count metadata is invalid");
  }
  const auto *n_counts = reinterpret_cast<const uint16_t *>(n_count_bytes.data());

  const size_t expected_packed_size =
      static_cast<size_t>((compressed.original_size + 3) / 4);
  if (packed_bases.size() != expected_packed_size) {
    throw std::runtime_error("Decoded packed basecalls have an unexpected size");
  }

  uint64_t total_n_count = 0;
  for (uint64_t i = 0; i < expected_block_count; ++i) {
    total_n_count += n_counts[i];
  }
  if (n_position_bytes.size() != total_n_count * sizeof(uint16_t)) {
    throw std::runtime_error("Decoded N-position payload has an unexpected size");
  }

  std::vector<uint8_t> basecalls(compressed.original_size);
  static constexpr uint8_t BASECALL_DECODE_TABLE[4] = {'A', 'C', 'G', 'T'};
  for (uint64_t index = 0; index < compressed.original_size; ++index) {
    const uint8_t packed_value = packed_bases[index / 4];
    const uint8_t code = static_cast<uint8_t>((packed_value >> (2 * (index % 4))) & 0x3);
    basecalls[index] = BASECALL_DECODE_TABLE[code];
  }

  const auto *n_positions =
      reinterpret_cast<const uint16_t *>(n_position_bytes.data());
  uint64_t n_offset = 0;
  for (uint64_t block_index = 0; block_index < expected_block_count; ++block_index) {
    const uint64_t block_start = block_index * compressed.n_block_size;
    const uint64_t block_end =
        std::min<uint64_t>(block_start + compressed.n_block_size,
                           compressed.original_size);
    for (uint16_t count = 0; count < n_counts[block_index]; ++count) {
      const uint16_t local_index = n_positions[n_offset++];
      if (block_start + local_index >= block_end) {
        throw std::runtime_error(
            "Decoded N-position metadata points outside its basecall block");
      }
      basecalls[block_start + local_index] = 'N';
    }
  }

  return basecalls;
}

FastqData rebuild_fastq(const std::vector<uint64_t> &line_offsets,
                        const std::vector<uint8_t> &identifiers,
                        const std::vector<uint8_t> &basecalls,
                        const std::vector<uint8_t> &quality_scores,
                        uint64_t num_records) {
  FastqData data;
  data.line_offsets = line_offsets;
  data.num_records = num_records;

  if (line_offsets.empty()) {
    throw std::runtime_error("Decoded line-offset metadata is empty");
  }

  const uint64_t file_size = line_offsets.back();
  data.raw_bytes.resize(file_size);

  uint64_t id_offset = 0;
  uint64_t seq_offset = 0;
  uint64_t qual_offset = 0;

  for (uint64_t record = 0; record < num_records; ++record) {
    const uint64_t id_line = 4 * record;
    const uint64_t seq_line = id_line + 1;
    const uint64_t plus_line = id_line + 2;
    const uint64_t qual_line = id_line + 3;

    const uint64_t id_start = line_offsets[id_line];
    const uint64_t seq_start = line_offsets[seq_line];
    const uint64_t plus_start = line_offsets[plus_line];
    const uint64_t qual_start = line_offsets[qual_line];
    const uint64_t next_start = line_offsets[qual_line + 1];

    const uint64_t id_len = seq_start - id_start - 2;
    const uint64_t seq_len = plus_start - seq_start - 1;
    const uint64_t plus_len = qual_start - plus_start - 1;
    const uint64_t qual_len = next_start - qual_start - 1;

    if (plus_len != 1) {
      throw std::runtime_error("Decoded line-offset metadata is incompatible "
                               "with ignored plus lines");
    }
    if (id_offset + id_len > identifiers.size() ||
        seq_offset + seq_len > basecalls.size() ||
        qual_offset + qual_len > quality_scores.size()) {
      throw std::runtime_error(
          "Decoded FASTQ field stream exceeds its uncompressed size");
    }

    data.raw_bytes[id_start] = '@';
    std::memcpy(data.raw_bytes.data() + id_start + 1,
                identifiers.data() + id_offset, id_len);
    data.raw_bytes[seq_start - 1] = '\n';

    std::memcpy(data.raw_bytes.data() + seq_start,
                basecalls.data() + seq_offset, seq_len);
    data.raw_bytes[plus_start - 1] = '\n';

    data.raw_bytes[plus_start] = '+';
    data.raw_bytes[qual_start - 1] = '\n';

    std::memcpy(data.raw_bytes.data() + qual_start,
                quality_scores.data() + qual_offset, qual_len);
    data.raw_bytes[next_start - 1] = '\n';

    id_offset += id_len;
    seq_offset += seq_len;
    qual_offset += qual_len;
  }

  if (id_offset != identifiers.size() || seq_offset != basecalls.size() ||
      qual_offset != quality_scores.size()) {
    throw std::runtime_error(
        "Decoded FASTQ field streams contain trailing bytes outside the index");
  }

  return data;
}

} // namespace

std::vector<uint8_t> gpu_compress(const std::vector<uint8_t> &input,
                                  size_t chunk_size) {
  if (input.empty()) {
    return {};
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint8_t *d_input = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input.size()));
  CUDA_CHECK(cudaMemcpyAsync(d_input, input.data(), input.size(),
                             cudaMemcpyHostToDevice, stream));

  std::vector<uint8_t> output =
      nvcomp_zstd_compress_device(d_input, input.size(), chunk_size, stream)
          .payload;

  cudaFree(d_input);
  cudaStreamDestroy(stream);
  return output;
}

std::vector<uint8_t> gpu_decompress(const std::vector<uint8_t> &compressed) {
  if (compressed.empty()) {
    return {};
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<uint8_t> output =
      nvcomp_zstd_decompress(ZstdCompressedBlock{compressed, compressed.size()},
                             NVCOMP_ZSTD_CHUNK_SIZE_DEFAULT, stream);
  cudaStreamDestroy(stream);
  return output;
}

CompressedFastqData compress_fastq(const FastqData &data, size_t chunk_size,
                                   const BscConfig &bsc_config) {
  const FastqFieldStats stats = compute_field_stats(data);
  const size_t field_slice_size = MAX_FIELD_SLICE_SIZE;

  CompressedFastqData result;
  result.num_records = data.num_records;
  result.identifiers.original_size = stats.identifiers_size;
  result.quality_scores.original_size = stats.quality_scores_size;
  result.line_lengths.original_size = stats.line_length_size;
  result.line_offset_count = data.line_offsets.size();
  const bool use_columnar_identifiers = data.identifier_layout.columnar;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint8_t *d_raw_bytes = nullptr;
  uint64_t *d_line_offsets = nullptr;
  uint64_t *d_identifier_lengths = nullptr;
  uint64_t *d_basecall_lengths = nullptr;
  uint64_t *d_quality_lengths = nullptr;
  uint64_t *d_identifier_offsets = nullptr;
  uint64_t *d_basecall_offsets = nullptr;
  uint64_t *d_quality_offsets = nullptr;
  uint32_t *d_line_lengths = nullptr;
  DeviceFieldBuffers fields;

  try {
    if (!data.raw_bytes.empty()) {
      CUDA_CHECK(cudaMalloc(&d_raw_bytes, data.raw_bytes.size()));
      CUDA_CHECK(cudaMemcpyAsync(d_raw_bytes, data.raw_bytes.data(),
                                 data.raw_bytes.size(), cudaMemcpyHostToDevice,
                                 stream));
    }

    if (!data.line_offsets.empty()) {
      CUDA_CHECK(cudaMalloc(&d_line_offsets,
                            data.line_offsets.size() * sizeof(uint64_t)));
      CUDA_CHECK(cudaMemcpyAsync(d_line_offsets, data.line_offsets.data(),
                                 data.line_offsets.size() * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, stream));
    }

    if (data.num_records > 0) {
      CUDA_CHECK(cudaMalloc(&d_identifier_lengths,
                            data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(
          cudaMalloc(&d_basecall_lengths, data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(
          cudaMalloc(&d_quality_lengths, data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(cudaMalloc(&d_identifier_offsets,
                            data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(
          cudaMalloc(&d_basecall_offsets, data.num_records * sizeof(uint64_t)));
      CUDA_CHECK(
          cudaMalloc(&d_quality_offsets, data.num_records * sizeof(uint64_t)));

      const uint32_t block_size = 256;
      const uint32_t grid_size = static_cast<uint32_t>(
          (data.num_records + block_size - 1) / block_size);

      compute_field_lengths_kernel<<<grid_size, block_size, 0, stream>>>(
          d_line_offsets, d_identifier_lengths, d_basecall_lengths,
          d_quality_lengths, data.num_records);
      CUDA_CHECK(cudaGetLastError());

      thrust::exclusive_scan(thrust::cuda::par.on(stream),
                             thrust::device_pointer_cast(d_identifier_lengths),
                             thrust::device_pointer_cast(d_identifier_lengths) +
                                 data.num_records,
                             thrust::device_pointer_cast(d_identifier_offsets));
      thrust::exclusive_scan(thrust::cuda::par.on(stream),
                             thrust::device_pointer_cast(d_basecall_lengths),
                             thrust::device_pointer_cast(d_basecall_lengths) +
                                 data.num_records,
                             thrust::device_pointer_cast(d_basecall_offsets));
      thrust::exclusive_scan(thrust::cuda::par.on(stream),
                             thrust::device_pointer_cast(d_quality_lengths),
                             thrust::device_pointer_cast(d_quality_lengths) +
                                 data.num_records,
                             thrust::device_pointer_cast(d_quality_offsets));

      if (!use_columnar_identifiers && stats.identifiers_size > 0) {
        CUDA_CHECK(cudaMalloc(&fields.identifiers, stats.identifiers_size));
      }
      if (stats.basecalls_size > 0) {
        CUDA_CHECK(cudaMalloc(&fields.basecalls, stats.basecalls_size));
      }
      if (stats.quality_scores_size > 0) {
        CUDA_CHECK(
            cudaMalloc(&fields.quality_scores, stats.quality_scores_size));
      }

      gather_fields_kernel<<<grid_size, block_size, 0, stream>>>(
          d_raw_bytes, d_line_offsets, d_identifier_offsets, d_basecall_offsets,
          d_quality_offsets, d_identifier_lengths, d_basecall_lengths,
          d_quality_lengths, fields.identifiers, fields.basecalls,
          fields.quality_scores, data.num_records);
      CUDA_CHECK(cudaGetLastError());
    }

    if (!data.line_offsets.empty()) {
      const uint64_t line_length_count = data.line_offsets.size();
      CUDA_CHECK(
          cudaMalloc(&d_line_lengths, line_length_count * sizeof(uint32_t)));
      delta_encode_offsets_to_lengths(d_line_offsets, d_line_lengths,
                                      line_length_count, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cerr << "Compressing identifiers (" << stats.identifiers_size
              << " bytes)..." << std::endl;
    if (use_columnar_identifiers) {
      result.identifiers = compress_identifiers_columnar(data, field_slice_size,
                                                         chunk_size);
      uint64_t compressed_identifier_size = result.identifiers.flat_data.payload.size();
      for (const auto &column : result.identifiers.columns) {
        compressed_identifier_size += column.values.payload.size();
        compressed_identifier_size += column.lengths.payload.size();
      }
      std::cerr << "  Mode: "
                << (result.identifiers.mode == IdentifierCompressionMode::Columnar
                        ? "columnar"
                        : "flat")
                << std::endl;
      if (result.identifiers.mode == IdentifierCompressionMode::Columnar) {
        std::cerr << "  Columns: " << result.identifiers.columns.size()
                  << std::endl;
        for (size_t i = 0; i < result.identifiers.columns.size(); ++i) {
          const auto &column = result.identifiers.columns[i];
          const uint64_t raw_value_size = column.values.original_size;
          const uint64_t comp_value_size = column.values.payload.size();
          const uint64_t raw_length_size = column.lengths.original_size;
          const uint64_t comp_length_size = column.lengths.payload.size();
          const uint64_t raw_total = raw_value_size + raw_length_size;
          const uint64_t comp_total = comp_value_size + comp_length_size;
          std::cerr << "    [" << i << "] "
                    << identifier_column_kind_name(column.kind)
                    << "/" << identifier_column_encoding_name(column.encoding)
                    << " values " << raw_value_size << " -> "
                    << comp_value_size << " B";
          if (column.kind == IdentifierColumnKind::String) {
            std::cerr << ", lengths " << raw_length_size << " -> "
                      << comp_length_size << " B";
          }
          std::cerr << ", total " << raw_total << " -> " << comp_total
                    << " B ("
                    << compression_ratio_percent(comp_total, raw_total) << " %)"
                    << std::endl;
        }
      }
      std::cerr << "  -> " << compressed_identifier_size << " bytes"
                << std::endl;
    } else {
      auto id_chunks =
          gpu_compress_device_chunked(fields.identifiers, stats.identifiers_size,
                                      field_slice_size, chunk_size, stream);
      result.identifiers.mode = IdentifierCompressionMode::Flat;
      result.identifiers.flat_data.original_size = stats.identifiers_size;
      result.identifiers.flat_data.payload = std::move(id_chunks.data);
      result.identifiers.compressed_flat_chunk_sizes =
          std::move(id_chunks.chunk_sizes);
      std::cerr << "  Mode: flat" << std::endl;
      std::cerr << "  -> " << result.identifiers.flat_data.payload.size()
                << " bytes" << std::endl;
    }

    std::cerr << "Compressing basecalls (" << stats.basecalls_size
              << " bytes)..." << std::endl;
    result.basecalls = compress_basecalls_device(fields.basecalls,
                                                 stats.basecalls_size,
                                                 field_slice_size, chunk_size,
                                                 stream);
    const uint64_t compressed_basecall_size =
        result.basecalls.packed_bases.payload.size() +
        result.basecalls.n_counts.payload.size() +
        result.basecalls.n_positions.payload.size();
    uint64_t total_n_count = 0;
    const uint64_t basecall_block_count =
        result.basecalls.original_size == 0
            ? 0
            : (result.basecalls.original_size + result.basecalls.n_block_size - 1) /
                  result.basecalls.n_block_size;
    if (result.basecalls.n_counts.original_size !=
        basecall_block_count * sizeof(uint16_t)) {
      throw std::runtime_error("Compressed N-count payload has an unexpected size");
    }
    if (result.basecalls.n_positions.original_size % sizeof(uint16_t) != 0) {
      throw std::runtime_error(
          "Compressed N-position payload has an unexpected size");
    }
    total_n_count = result.basecalls.n_positions.original_size / sizeof(uint16_t);
    std::cerr << "  Packed bases: " << result.basecalls.packed_bases.original_size
              << " bytes, N positions: " << total_n_count << std::endl;
    std::cerr << "  -> " << compressed_basecall_size << " bytes" << std::endl;

    std::cerr << "Compressing quality scores (" << stats.quality_scores_size
              << " bytes)..." << std::endl;
    const size_t quality_chunk_count =
        stats.quality_scores_size == 0
            ? 0
            : (stats.quality_scores_size + BSC_QUALITY_CHUNK_SIZE - 1) /
                  BSC_QUALITY_CHUNK_SIZE;
    const auto resolved_bsc =
        resolve_bsc_config(bsc_config, quality_chunk_count);
    std::cerr << "  BSC backend: " << bsc_backend_name(resolved_bsc.backend)
              << ", jobs: " << resolved_bsc.parallelism
              << ", chunks: " << quality_chunk_count << std::endl;
    std::vector<uint8_t> h_quality_scores(stats.quality_scores_size);
    if (stats.quality_scores_size > 0) {
      CUDA_CHECK(cudaMemcpyAsync(h_quality_scores.data(), fields.quality_scores,
                                 stats.quality_scores_size,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto qual_chunks = bsc_compress_chunked(h_quality_scores.data(),
                                            stats.quality_scores_size,
                                            BSC_QUALITY_CHUNK_SIZE,
                                            bsc_config);
    result.quality_scores.payload = std::move(qual_chunks.data);
    result.compressed_quality_chunk_sizes =
        std::move(qual_chunks.compressed_chunk_sizes);
    result.uncompressed_quality_chunk_sizes =
        std::move(qual_chunks.uncompressed_chunk_sizes);
    std::cerr << "  -> " << result.quality_scores.payload.size() << " bytes"
              << std::endl;

    std::cerr << "Compressing line lengths (" << stats.line_length_size
              << " bytes)..." << std::endl;
    auto index_chunks = gpu_compress_device_chunked(
        reinterpret_cast<const uint8_t *>(d_line_lengths),
        stats.line_length_size, field_slice_size, chunk_size, stream);
    result.line_lengths.payload = std::move(index_chunks.data);
    result.compressed_line_length_chunk_sizes =
        std::move(index_chunks.chunk_sizes);
    std::cerr << "  -> " << result.line_lengths.payload.size() << " bytes"
              << std::endl;
  } catch (...) {
    cuda_free_if_set(d_line_lengths);
    cuda_free_if_set(fields.identifiers);
    cuda_free_if_set(fields.basecalls);
    cuda_free_if_set(fields.quality_scores);
    cuda_free_if_set(d_quality_offsets);
    cuda_free_if_set(d_basecall_offsets);
    cuda_free_if_set(d_identifier_offsets);
    cuda_free_if_set(d_quality_lengths);
    cuda_free_if_set(d_basecall_lengths);
    cuda_free_if_set(d_identifier_lengths);
    cuda_free_if_set(d_line_offsets);
    cuda_free_if_set(d_raw_bytes);
    cudaStreamDestroy(stream);
    throw;
  }

  cuda_free_if_set(d_line_lengths);
  cuda_free_if_set(fields.identifiers);
  cuda_free_if_set(fields.basecalls);
  cuda_free_if_set(fields.quality_scores);
  cuda_free_if_set(d_quality_offsets);
  cuda_free_if_set(d_basecall_offsets);
  cuda_free_if_set(d_identifier_offsets);
  cuda_free_if_set(d_quality_lengths);
  cuda_free_if_set(d_basecall_lengths);
  cuda_free_if_set(d_identifier_lengths);
  cuda_free_if_set(d_line_offsets);
  cuda_free_if_set(d_raw_bytes);
  cudaStreamDestroy(stream);
  return result;
}

FastqData decompress_fastq(const CompressedFastqData &compressed,
                           const BscConfig &bsc_config) {
  std::cerr << "Decompressing identifiers..." << std::endl;
  const auto identifiers =
      decompress_identifiers(compressed.identifiers, compressed.num_records);
  std::cerr << "  -> " << identifiers.size() << " bytes" << std::endl;

  std::cerr << "Decompressing basecalls..." << std::endl;
  const auto n_count_bytes = nvcomp_zstd_decompress(compressed.basecalls.n_counts);
  const auto packed_bases = gpu_decompress_chunked(
      compressed.basecalls.packed_bases.payload,
      compressed.basecalls.compressed_packed_chunk_sizes,
      compressed.basecalls.packed_bases.original_size);
  const auto n_position_bytes = gpu_decompress_chunked(
      compressed.basecalls.n_positions.payload,
      compressed.basecalls.compressed_n_position_chunk_sizes,
      compressed.basecalls.n_positions.original_size);
  const auto basecalls = decode_basecalls(compressed.basecalls, n_count_bytes,
                                          packed_bases, n_position_bytes);
  std::cerr << "  -> " << basecalls.size() << " bytes" << std::endl;

  std::cerr << "Decompressing quality scores..." << std::endl;
  const auto resolved_bsc = resolve_bsc_config(
      bsc_config, compressed.compressed_quality_chunk_sizes.size());
  std::cerr << "  BSC backend: " << bsc_backend_name(resolved_bsc.backend)
            << ", jobs: " << resolved_bsc.parallelism
            << ", chunks: " << compressed.compressed_quality_chunk_sizes.size()
            << std::endl;
  const auto quality_scores = bsc_decompress_chunked(
      compressed.quality_scores.payload,
      compressed.compressed_quality_chunk_sizes,
      compressed.uncompressed_quality_chunk_sizes,
      compressed.quality_scores.original_size, bsc_config);
  std::cerr << "  -> " << quality_scores.size() << " bytes" << std::endl;

  std::cerr << "Decompressing line lengths..." << std::endl;
  const auto line_offset_bytes =
      gpu_decompress_chunked(compressed.line_lengths.payload,
                             compressed.compressed_line_length_chunk_sizes,
                             compressed.line_lengths.original_size);
  std::cerr << "  -> " << line_offset_bytes.size() << " bytes" << std::endl;

  if (compressed.line_offset_count == 0) {
    throw std::runtime_error("Decoded line-offset count is invalid");
  }
  if (line_offset_bytes.size() !=
      compressed.line_offset_count * sizeof(uint32_t)) {
    throw std::runtime_error(
        "Decoded line-length payload has an unexpected size");
  }

  std::vector<uint64_t> line_offsets(compressed.line_offset_count);
  uint32_t *d_line_lengths = nullptr;
  uint64_t *d_line_offsets = nullptr;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  try {
    CUDA_CHECK(cudaMalloc(&d_line_lengths, line_offset_bytes.size()));
    CUDA_CHECK(
        cudaMalloc(&d_line_offsets, line_offsets.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_line_lengths, line_offset_bytes.data(),
                               line_offset_bytes.size(), cudaMemcpyHostToDevice,
                               stream));
    delta_decode_lengths_to_offsets(d_line_lengths, d_line_offsets,
                                    line_offsets.size(), stream);
    CUDA_CHECK(cudaMemcpyAsync(line_offsets.data(), d_line_offsets,
                               line_offsets.size() * sizeof(uint64_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } catch (...) {
    cuda_free_if_set(d_line_offsets);
    cuda_free_if_set(d_line_lengths);
    cudaStreamDestroy(stream);
    throw;
  }

  cuda_free_if_set(d_line_offsets);
  cuda_free_if_set(d_line_lengths);
  cudaStreamDestroy(stream);
  return rebuild_fastq(line_offsets, identifiers, basecalls, quality_scores,
                       compressed.num_records);
}

} // namespace gpufastq
