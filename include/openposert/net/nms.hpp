#pragma once

#include <array>

namespace openposert {

template <typename T>
void nms_gpu(T* target_ptr, int* kernel_ptr, const T* const source_ptr,
             const T threshold, const std::array<int, 4>& target_size,
             const std::array<int, 4>& source_size, const T offset_x,
             const T offset_y);

}  // namespace openposert
