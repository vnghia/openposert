#pragma once

#include <cstddef>
#include <string>

#include "cuda_runtime.h"
#include "openposert/core/common.hpp"

namespace openposert {

static constexpr auto CUDA_NUM_THREADS = 512u;

inline unsigned int get_number_cuda_blocks(
    const unsigned int total_required,
    const unsigned int number_cuda_threads = CUDA_NUM_THREADS) {
  return (total_required + number_cuda_threads - 1) / number_cuda_threads;
}

inline void cuda_check(const int line = -1, const std::string& function = "",
                       const std::string& file = "") {
  const auto error_code = cudaPeekAtLastError();
  if (error_code != cudaSuccess)
    error("Cuda check failed (" + std::to_string(error_code) + " vs. " +
              std::to_string(cudaSuccess) +
              "): " + cudaGetErrorString(error_code),
          line, function, file);
}

}  // namespace openposert
