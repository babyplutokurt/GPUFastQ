#include "gpufastq/compression_workflow.hpp"

#include "gpufastq/fastq_parser.hpp"
#include "gpufastq/gpu_compressor.hpp"
#include "gpufastq/serializer.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace gpufastq::workflow {

namespace {

long long
elapsed_ms(const std::chrono::high_resolution_clock::time_point &start,
           const std::chrono::high_resolution_clock::time_point &end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

} // namespace

int compress(const std::string &input_path, const std::string &output_path) {
  using clock = std::chrono::high_resolution_clock;
  const auto t0 = clock::now();

  std::cerr << "=== Parsing FASTQ: " << input_path << " ===\n";
  const auto data = parse_fastq(input_path);
  const auto stats = compute_field_stats(data);
  const auto t1 = clock::now();

  std::cerr << "Parsed " << data.num_records << " records\n"
            << "  FASTQ bytes:     " << data.raw_bytes.size() << " B\n"
            << "  Line offsets:    " << data.line_offsets.size() << " B\n"
            << "  Identifiers:     " << stats.identifiers_size << " B\n"
            << "  Basecalls:       " << stats.basecalls_size << " B\n"
            << "  Quality scores:  " << stats.quality_scores_size << " B\n"
            << "  Line lengths:    " << stats.line_length_size << " B\n";

  const size_t raw_payload_size =
      stats.identifiers_size + stats.basecalls_size +
      stats.quality_scores_size + stats.line_length_size;

  std::cerr << "\n=== GPU Compression ===\n";
  const auto compressed = compress_fastq(data);
  const auto t2 = clock::now();

  const size_t compressed_payload_size =
      compressed.identifiers.payload.size() +
      compressed.basecalls.payload.size() +
      compressed.quality_scores.payload.size() +
      compressed.line_lengths.payload.size();

  std::cerr << "\n=== Serializing: " << output_path << " ===\n";
  serialize(output_path, compressed);
  const auto t3 = clock::now();

  const size_t input_size = fs::file_size(input_path);
  const size_t output_size = fs::file_size(output_path);

  std::cerr << "\n=== Summary ===\n"
            << "  Input file:        " << input_size << " B\n"
            << "  Output file:       " << output_size << " B\n"
            << "  File ratio:        " << 100.0 * output_size / input_size
            << " %\n"
            << "  Raw payload:       " << raw_payload_size << " B\n"
            << "  Compressed payload:" << compressed_payload_size << " B\n"
            << "  Payload ratio:     "
            << 100.0 * compressed_payload_size / raw_payload_size << " %\n"
            << "  Parse time:        " << elapsed_ms(t0, t1) << " ms\n"
            << "  Compress time:     " << elapsed_ms(t1, t2) << " ms\n"
            << "  Serialize time:    " << elapsed_ms(t2, t3) << " ms\n"
            << "  Total time:        " << elapsed_ms(t0, t3) << " ms\n";
  return 0;
}

int roundtrip(const std::string &input_path) {
  std::cerr << "=== Round-trip verification ===\n";

  const auto original = parse_fastq(input_path);
  std::cerr << "Parsed " << original.num_records << " records\n";

  const auto compressed = compress_fastq(original);

  const std::string tmp_path = input_path + ".gpufq.tmp";
  serialize(tmp_path, compressed);
  const auto loaded = deserialize(tmp_path);
  const auto decoded = decompress_fastq(loaded);

  bool ok = true;
  if (original.num_records != decoded.num_records) {
    std::cerr << "FAIL: record count\n";
    ok = false;
  }
  if (original.raw_bytes != decoded.raw_bytes) {
    std::cerr << "FAIL: raw FASTQ bytes\n";
    ok = false;
  }
  if (original.line_offsets != decoded.line_offsets) {
    std::cerr << "FAIL: line offsets\n";
    ok = false;
  }

  fs::remove(tmp_path);

  std::cerr << (ok ? "\nROUND-TRIP: PASSED ✓\n" : "\nROUND-TRIP: FAILED ✗\n");
  return ok ? 0 : 1;
}

} // namespace gpufastq::workflow
