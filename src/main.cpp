#include "gpufastq/fastq_parser.hpp"
#include "gpufastq/gpu_compressor.hpp"
#include "gpufastq/serializer.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

void print_usage(const char *prog) {
  std::cerr << "GPUFastQ — GPU-accelerated FASTQ compression (nvcomp zstd)\n\n"
            << "Usage:\n"
            << "  " << prog << " compress   <input.fastq> <output.gpufq>\n"
            << "  " << prog << " decompress <input.gpufq> <output.fastq>\n"
            << "  " << prog << " roundtrip  <input.fastq>\n"
            << std::endl;
}

// ----------------------------------------------------------------

int do_compress(const std::string &in, const std::string &out) {
  using clk = std::chrono::high_resolution_clock;
  auto t0 = clk::now();

  std::cerr << "=== Parsing FASTQ: " << in << " ===\n";
  auto data = gpufastq::parse_fastq(in);
  auto t1 = clk::now();

  std::cerr << "Parsed " << data.num_records << " records\n"
            << "  Identifiers:    " << data.identifiers.size() << " B\n"
            << "  Basecalls:      " << data.basecalls.size() << " B\n"
            << "  Quality scores: " << data.quality_scores.size() << " B\n";

  size_t raw = data.identifiers.size() + data.basecalls.size() +
               data.quality_scores.size();

  std::cerr << "\n=== GPU Compression ===\n";
  auto comp = gpufastq::compress_fastq(data);
  auto t2 = clk::now();

  size_t cmp = comp.compressed_identifiers.size() +
               comp.compressed_basecalls.size() +
               comp.compressed_quality.size();

  std::cerr << "\n=== Serializing: " << out << " ===\n";
  gpufastq::serialize(out, comp);
  auto t3 = clk::now();

  auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };

  size_t in_sz = fs::file_size(in);
  size_t out_sz = fs::file_size(out);

  std::cerr << "\n=== Summary ===\n"
            << "  Input file:        " << in_sz << " B\n"
            << "  Output file:       " << out_sz << " B\n"
            << "  File ratio:        " << 100.0 * out_sz / in_sz << " %\n"
            << "  Raw payload:       " << raw << " B\n"
            << "  Compressed payload:" << cmp << " B\n"
            << "  Payload ratio:     " << 100.0 * cmp / raw << " %\n"
            << "  Parse time:        " << ms(t0, t1) << " ms\n"
            << "  Compress time:     " << ms(t1, t2) << " ms\n"
            << "  Serialize time:    " << ms(t2, t3) << " ms\n"
            << "  Total time:        " << ms(t0, t3) << " ms\n";
  return 0;
}

// ----------------------------------------------------------------

int do_decompress(const std::string &in, const std::string &out) {
  using clk = std::chrono::high_resolution_clock;
  auto t0 = clk::now();

  std::cerr << "=== Deserializing: " << in << " ===\n";
  auto comp = gpufastq::deserialize(in);
  auto t1 = clk::now();

  std::cerr << "Records: " << comp.num_records << "\n";

  std::cerr << "\n=== GPU Decompression ===\n";
  auto data = gpufastq::decompress_fastq(comp);
  auto t2 = clk::now();

  std::cerr << "\n=== Writing FASTQ: " << out << " ===\n";
  gpufastq::write_fastq(out, data);
  auto t3 = clk::now();

  auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };

  std::cerr << "\n=== Summary ===\n"
            << "  Records:          " << data.num_records << "\n"
            << "  Deserialize time: " << ms(t0, t1) << " ms\n"
            << "  Decompress time:  " << ms(t1, t2) << " ms\n"
            << "  Write time:       " << ms(t2, t3) << " ms\n"
            << "  Total time:       " << ms(t0, t3) << " ms\n";
  return 0;
}

// ----------------------------------------------------------------

int do_roundtrip(const std::string &input_path) {
  std::cerr << "=== Round-trip verification ===\n";

  auto original = gpufastq::parse_fastq(input_path);
  std::cerr << "Parsed " << original.num_records << " records\n";

  auto compressed = gpufastq::compress_fastq(original);

  std::string tmp = input_path + ".gpufq.tmp";
  gpufastq::serialize(tmp, compressed);
  auto loaded = gpufastq::deserialize(tmp);
  auto decoded = gpufastq::decompress_fastq(loaded);

  bool ok = true;
  if (original.num_records != decoded.num_records) {
    std::cerr << "FAIL: record count\n";
    ok = false;
  }
  if (original.identifiers != decoded.identifiers) {
    std::cerr << "FAIL: identifiers\n";
    ok = false;
  }
  if (original.basecalls != decoded.basecalls) {
    std::cerr << "FAIL: basecalls\n";
    ok = false;
  }
  if (original.quality_scores != decoded.quality_scores) {
    std::cerr << "FAIL: quality scores\n";
    ok = false;
  }

  fs::remove(tmp);

  std::cerr << (ok ? "\nROUND-TRIP: PASSED ✓\n" : "\nROUND-TRIP: FAILED ✗\n");
  return ok ? 0 : 1;
}

// ----------------------------------------------------------------

int main(int argc, char *argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  std::string cmd = argv[1];
  try {
    if (cmd == "compress" && argc >= 4)
      return do_compress(argv[2], argv[3]);
    if (cmd == "decompress" && argc >= 4)
      return do_decompress(argv[2], argv[3]);
    if (cmd == "roundtrip" && argc >= 3)
      return do_roundtrip(argv[2]);

    print_usage(argv[0]);
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
