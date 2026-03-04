#include "gpufastq/decompression_workflow.hpp"

#include "gpufastq/fastq_parser.hpp"
#include "gpufastq/gpu_compressor.hpp"
#include "gpufastq/serializer.hpp"

#include <chrono>
#include <iostream>

namespace gpufastq::workflow {

namespace {

long long elapsed_ms(const std::chrono::high_resolution_clock::time_point &start,
                     const std::chrono::high_resolution_clock::time_point &end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

} // namespace

int decompress(const std::string &input_path, const std::string &output_path,
               const BscConfig &bsc_config) {
  using clock = std::chrono::high_resolution_clock;
  const auto t0 = clock::now();

  std::cerr << "=== Deserializing: " << input_path << " ===\n";
  const auto compressed = deserialize(input_path);
  const auto t1 = clock::now();

  std::cerr << "Records: " << compressed.num_records << "\n";

  std::cerr << "\n=== GPU Decompression ===\n";
  if (bsc_config.backend != BscBackend::Default) {
    std::cerr << "BSC backend:       " << bsc_backend_name(bsc_config.backend)
              << "\n";
  }
  const auto data = decompress_fastq(compressed, bsc_config);
  const auto t2 = clock::now();

  std::cerr << "\n=== Writing FASTQ: " << output_path << " ===\n";
  write_fastq(output_path, data);
  const auto t3 = clock::now();

  std::cerr << "\n=== Summary ===\n"
            << "  Records:          " << data.num_records << "\n"
            << "  Deserialize time: " << elapsed_ms(t0, t1) << " ms\n"
            << "  Decompress time:  " << elapsed_ms(t1, t2) << " ms\n"
            << "  Write time:       " << elapsed_ms(t2, t3) << " ms\n"
            << "  Total time:       " << elapsed_ms(t0, t3) << " ms\n";
  return 0;
}

} // namespace gpufastq::workflow
