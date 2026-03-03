#pragma once

#include <string>

namespace gpufastq::workflow {

int compress(const std::string &input_path, const std::string &output_path);
int roundtrip(const std::string &input_path);

} // namespace gpufastq::workflow
