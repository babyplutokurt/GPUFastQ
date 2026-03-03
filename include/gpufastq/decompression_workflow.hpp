#pragma once

#include <string>

namespace gpufastq::workflow {

int decompress(const std::string &input_path, const std::string &output_path);

} // namespace gpufastq::workflow
