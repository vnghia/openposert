#pragma once

#include <algorithm>

#include "cuda_fp16.h"
#include "cuda_runtime.h"

namespace openposert {

inline __device__ void cubic_sequential_data(int* x_int_array, int* y_int_array,
                                             __half& dx, __half& dy,
                                             const __half x_source,
                                             const __half y_source,
                                             const int width_source,
                                             const int height_source) {
  x_int_array[1] =
      std::clamp(static_cast<int>(hfloor(x_source)), 0, width_source - 1);
  x_int_array[0] = std::max(0, x_int_array[1] - 1);
  x_int_array[2] = std::min(width_source - 1, x_int_array[1] + 1);
  x_int_array[3] = std::min(width_source - 1, x_int_array[2] + 1);
  dx = x_source - static_cast<__half>(x_int_array[1]);

  y_int_array[1] =
      std::clamp(static_cast<int>(hfloor(y_source)), 0, height_source - 1);
  y_int_array[0] = std::max(0, y_int_array[1] - 1);
  y_int_array[2] = std::min(height_source - 1, y_int_array[1] + 1);
  y_int_array[3] = std::min(height_source - 1, y_int_array[2] + 1);
  dy = y_source - static_cast<__half>(y_int_array[1]);
}

inline __device__ __half cubic_interpolate(const __half v0, const __half v1,
                                           const __half v2, const __half v3,
                                           const __half dx) {
  // http://www.paulinternet.nl/?page=bicubic
  return (static_cast<__half>(-0.5f) * v0 + static_cast<__half>(1.5f) * v1 -
          static_cast<__half>(1.5f) * v2 + static_cast<__half>(0.5f) * v3) *
             dx * dx * dx +
         (v0 - static_cast<__half>(2.5f) * v1 + static_cast<__half>(2.f) * v2 -
          static_cast<__half>(0.5f) * v3) *
             dx * dx -
         static_cast<__half>(0.5f) * (v0 - v2) * dx + v1;
}

inline __device__ __half bicubic_interpolate(const __half* const source_ptr,
                                             const __half x_source,
                                             const __half y_source,
                                             const int width_source,
                                             const int height_source,
                                             const int width_source_ptr) {
  int x_int_array[4];
  int y_int_array[4];
  __half dx;
  __half dy;
  cubic_sequential_data(x_int_array, y_int_array, dx, dy, x_source, y_source,
                        width_source, height_source);

  __half temp[4];
  for (unsigned char i = 0; i < 4; i++) {
    const auto offset = y_int_array[i] * width_source_ptr;
    temp[i] = cubic_interpolate(source_ptr[offset + x_int_array[0]],
                                source_ptr[offset + x_int_array[1]],
                                source_ptr[offset + x_int_array[2]],
                                source_ptr[offset + x_int_array[3]], dx);
  }
  return cubic_interpolate(temp[0], temp[1], temp[2], temp[3], dy);
}

}  // namespace openposert
