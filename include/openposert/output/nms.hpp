#pragma once

#include <array>

namespace openposert {

void nms(float* target_ptr, int* kernel_ptr, const float* const source_ptr,
         const float threshold, const std::array<int, 4>& target_size,
         const std::array<int, 4>& source_size, const float offset_x,
         const float offset_y);

}  // namespace openposert
