#pragma once

#include "cuda_runtime.h"

namespace openposert {

// VERY IMPORTANT: these fast functions does not work for negative integer
// numbers. e.g., positive-int-round(-180.f) = -179.

// round functions
// signed

template <typename T>
inline __device__ char positive_char_round_cuda(const T a) {
  return char(a + 0.5f);
}

template <typename T>
inline __device__ signed char positive_s_char_round_cuda(const T a) {
  return (signed char)(a + 0.5f);
}

template <typename T>
inline __device__ int positive_int_round_cuda(const T a) {
  return int(a + 0.5f);
}

template <typename T>
inline __device__ long positive_long_round_cuda(const T a) {
  return long(a + 0.5f);
}

template <typename T>
inline __device__ long long positive_long_long_round_cuda(const T a) {
  return (long long)(a + 0.5f);
}

// unsigned
template <typename T>
inline __device__ unsigned char u_char_round_cuda(const T a) {
  return (unsigned char)(a + 0.5f);
}

template <typename T>
inline __device__ unsigned int u_int_round_cuda(const T a) {
  return (unsigned int)(a + 0.5f);
}

template <typename T>
inline __device__ unsigned long u_long_round_cuda(const T a) {
  return (unsigned long)(a + 0.5f);
}

template <typename T>
inline __device__ unsigned long long u_long_long_round_cuda(const T a) {
  return (unsigned long long)(a + 0.5f);
}

// max/min functions
template <class T>
inline __device__ T fast_max_cuda(const T a, const T b) {
  return (a > b ? a : b);
}

template <class T>
inline __device__ T fast_min_cuda(const T a, const T b) {
  return (a < b ? a : b);
}

template <class T>
inline __device__ T fast_truncate_cuda(const T value, const T min = 0,
                                       const T max = 1) {
  return fast_min_cuda(max, fast_max_cuda(min, value));
}

// cubic interpolation
template <typename T>
inline __device__ void cubic_sequential_data(int* x_int_array, int* y_int_array,
                                             T& dx, T& dy, const T x_source,
                                             const T y_source,
                                             const int width_source,
                                             const int height_source) {
  x_int_array[1] =
      fast_truncate_cuda(int(floor(x_source)), 0, width_source - 1);
  x_int_array[0] = fast_max_cuda(0, x_int_array[1] - 1);
  x_int_array[2] = fast_min_cuda(width_source - 1, x_int_array[1] + 1);
  x_int_array[3] = fast_min_cuda(width_source - 1, x_int_array[2] + 1);
  dx = x_source - x_int_array[1];

  y_int_array[1] =
      fast_truncate_cuda(int(floor(y_source)), 0, height_source - 1);
  y_int_array[0] = fast_max_cuda(0, y_int_array[1] - 1);
  y_int_array[2] = fast_min_cuda(height_source - 1, y_int_array[1] + 1);
  y_int_array[3] = fast_min_cuda(height_source - 1, y_int_array[2] + 1);
  dy = y_source - y_int_array[1];
}

template <typename T>
inline __device__ T cubic_interpolate(const T v0, const T v1, const T v2,
                                      const T v3, const T dx) {
  // http://www.paulinternet.nl/?page=bicubic
  return (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3) * dx * dx * dx +
         (v0 - 2.5f * v1 + 2.f * v2 - 0.5f * v3) * dx * dx -
         0.5f * (v0 - v2) * dx  // + (-0.5f * v0 + 0.5f * v2) * dx
         + v1;
}

template <typename T>
inline __device__ T bicubic_interpolate(const T* const source_ptr,
                                        const T x_source, const T y_source,
                                        const int width_source,
                                        const int height_source,
                                        const int width_source_ptr) {
  int x_int_array[4];
  int y_int_array[4];
  T dx;
  T dy;
  cubic_sequential_data(x_int_array, y_int_array, dx, dy, x_source, y_source,
                        width_source, height_source);

  T temp[4];
  for (unsigned char i = 0; i < 4; i++) {
    const auto offset = y_int_array[i] * width_source_ptr;
    temp[i] = cubic_interpolate(source_ptr[offset + x_int_array[0]],
                                source_ptr[offset + x_int_array[1]],
                                source_ptr[offset + x_int_array[2]],
                                source_ptr[offset + x_int_array[3]], dx);
  }
  return cubic_interpolate(temp[0], temp[1], temp[2], temp[3], dy);
}

template <typename T>
inline __device__ T bicubic_interpolate(const unsigned char* const source_ptr,
                                        const T x_source, const T y_source,
                                        const int width_source,
                                        const int height_source,
                                        const int width_source_ptr) {
  int x_int_array[4];
  int y_int_array[4];
  T dx;
  T dy;
  cubic_sequential_data(x_int_array, y_int_array, dx, dy, x_source, y_source,
                        width_source, height_source);

  T temp[4];
  for (unsigned char i = 0; i < 4; i++) {
    const auto offset = y_int_array[i] * width_source_ptr;
    temp[i] = cubic_interpolate(T(source_ptr[offset + x_int_array[0]]),
                                T(source_ptr[offset + x_int_array[1]]),
                                T(source_ptr[offset + x_int_array[2]]),
                                T(source_ptr[offset + x_int_array[3]]), dx);
  }
  return cubic_interpolate(temp[0], temp[1], temp[2], temp[3], dy);
}

template <typename T>
inline __device__ T bicubic_interpolate8_times(
    const T* const source_ptr, const T x_source, const T y_source,
    const int width_source, const int height_source, const int thread_idx_x,
    const int thread_idx_y) {
  // now we only need dx and dy
  const T dx =
      x_source - fast_truncate_cuda(int(floor(x_source)), 0, width_source - 1);
  const T dy =
      y_source - fast_truncate_cuda(int(floor(y_source)), 0, height_source - 1);

  T temp[4];
  for (unsigned char i = 0; i < 4; i++) {
    const auto offset =
        5 * (i + (thread_idx_y > 3 ? 1 : 0)) + (thread_idx_x > 3 ? 1 : 0);
    temp[i] =
        cubic_interpolate(source_ptr[offset], source_ptr[offset + 1],
                          source_ptr[offset + 2], source_ptr[offset + 3], dx);
  }
  return cubic_interpolate(temp[0], temp[1], temp[2], temp[3], dy);
}

template <typename T>
inline __device__ T add_weighted(const T value1, const T value2,
                                 const T alpha_value2) {
  return (1.f - alpha_value2) * value1 + alpha_value2 * value2;
}

template <typename T>
inline __device__ void add_color_weighted(T& color_r, T& color_g, T& color_b,
                                          const T* const color_to_add,
                                          const T alpha_color_to_add) {
  color_r = add_weighted(color_r, color_to_add[0], alpha_color_to_add);
  color_g = add_weighted(color_g, color_to_add[1], alpha_color_to_add);
  color_b = add_weighted(color_b, color_to_add[2], alpha_color_to_add);
}

}  // namespace openposert
