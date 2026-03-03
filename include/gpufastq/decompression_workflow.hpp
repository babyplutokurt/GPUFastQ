#pragma once

#include <string>

namespace gpufastq::workflow {

int decompress(const std::string &input_path, const std::string &output_path,
               size_t bsc_threads = 0);

} // namespace gpufastq::workflow
