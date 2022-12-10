#pragma once

#include <array>

#include "cuda_fp16.h"

namespace openposert {

void nms(__half* target_ptr, int* kernel_ptr, const __half* const source_ptr,
         const __half threshold, const std::array<int, 4>& target_size,
         const std::array<int, 4>& source_size, const __half offset_x,
         const __half offset_y);

}  // namespace openposert
