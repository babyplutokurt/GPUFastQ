#include "gpufastq/compression_workflow.hpp"
#include "gpufastq/decompression_workflow.hpp"

#include <iostream>
#include <limits>
#include <optional>
#include <string>

namespace {

std::optional<size_t> parse_bsc_threads_arg(const std::string &value) {
  size_t pos = 0;
  const unsigned long long parsed = std::stoull(value, &pos, 10);
  if (pos != value.size() || parsed == 0 ||
      parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error("--bsc-threads must be a positive integer");
  }
  return static_cast<size_t>(parsed);
}

} // namespace

void print_usage(const char *prog) {
  std::cerr << "GPUFastQ — GPU-accelerated FASTQ compression (nvcomp zstd)\n\n"
            << "Usage:\n"
            << "  " << prog
            << " compress   [--bsc-threads N] <input.fastq> <output.gpufq>\n"
            << "  " << prog
            << " decompress [--bsc-threads N] <input.gpufq> <output.fastq>\n"
            << "  " << prog
            << " roundtrip  [--bsc-threads N] <input.fastq>\n\n"
            << "Options:\n"
            << "  --bsc-threads N    Override CPU worker count for libbsc quality chunks\n\n"
            << "Environment:\n"
            << "  GPUFASTQ_BSC_THREADS  Default CPU worker count when --bsc-threads is not set\n"
            << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  std::string cmd = argv[1];
  try {
    size_t bsc_threads = 0;
    int argi = 2;
    while (argi < argc) {
      const std::string arg = argv[argi];
      if (arg != "--bsc-threads") {
        break;
      }
      if (argi + 1 >= argc) {
        throw std::runtime_error("Missing value for --bsc-threads");
      }
      bsc_threads = *parse_bsc_threads_arg(argv[argi + 1]);
      argi += 2;
    }

    if (cmd == "compress" && argc - argi >= 2)
      return gpufastq::workflow::compress(argv[argi], argv[argi + 1],
                                          bsc_threads);
    if (cmd == "decompress" && argc - argi >= 2)
      return gpufastq::workflow::decompress(argv[argi], argv[argi + 1],
                                            bsc_threads);
    if (cmd == "roundtrip" && argc - argi >= 1)
      return gpufastq::workflow::roundtrip(argv[argi], bsc_threads);

    print_usage(argv[0]);
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
