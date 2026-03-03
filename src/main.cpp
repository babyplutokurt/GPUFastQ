#include "gpufastq/compression_workflow.hpp"
#include "gpufastq/decompression_workflow.hpp"

#include <iostream>
#include <string>

void print_usage(const char *prog) {
  std::cerr << "GPUFastQ — GPU-accelerated FASTQ compression (nvcomp zstd)\n\n"
            << "Usage:\n"
            << "  " << prog << " compress   <input.fastq> <output.gpufq>\n"
            << "  " << prog << " decompress <input.gpufq> <output.fastq>\n"
            << "  " << prog << " roundtrip  <input.fastq>\n"
            << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  std::string cmd = argv[1];
  try {
    if (cmd == "compress" && argc >= 4)
      return gpufastq::workflow::compress(argv[2], argv[3]);
    if (cmd == "decompress" && argc >= 4)
      return gpufastq::workflow::decompress(argv[2], argv[3]);
    if (cmd == "roundtrip" && argc >= 3)
      return gpufastq::workflow::roundtrip(argv[2]);

    print_usage(argv[0]);
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
