#pragma once

#include <cstddef>
#include <string>

namespace openposert {

static constexpr auto CUDA_NUM_THREADS = 512u;

inline unsigned int get_number_cuda_blocks(
    const unsigned int total_required,
    const unsigned int number_cuda_threads = CUDA_NUM_THREADS) {
  return (total_required + number_cuda_threads - 1) / number_cuda_threads;
}

}  // namespace openposert
