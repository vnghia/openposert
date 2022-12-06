#pragma once

#include <array>
#include <string>
#include <vector>

namespace openposert {

template <typename T>
__global__ void resize_kernel(T* target_ptr, const T* const source_ptr,
                              const int width_source, const int height_source,
                              const int width_target, const int height_target);

template <typename T>
void resize_and_pad_rbg_gpu(T* target_ptr, const T* const src_ptr,
                            const int source_width, const int source_height,
                            const int target_width, const int target_height,
                            const T scale_factor);

template <typename T>
void resize_and_pad_rbg_gpu(T* target_ptr, const unsigned char* const src_ptr,
                            const int source_width, const int source_height,
                            const int target_width, const int target_height,
                            const T scale_factor);

}  // namespace openposert
