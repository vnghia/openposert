#pragma once

#include <string>

#include "spdlog/spdlog.h"

namespace openposert {

inline void error(const std::string& content, int line,
                  const std::string& function, const std::string& file) {
  spdlog::error("{} in {} at {}:{}", content, function, file, line);
}

}  // namespace openposert
