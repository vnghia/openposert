#pragma once

#include <array>
#include <string>
#include <vector>

namespace openposert {

template <typename T>
void resize_and_merge_cpu(T* target_ptr,
                          const std::vector<const T*>& source_ptrs,
                          const std::array<int, 4>& target_size,
                          const std::vector<std::array<int, 4>>& source_sizes,
                          const std::vector<T>& scale_input_to_net_inputs = {
                              1.f});

template <typename T>
void resize_and_merge_gpu(T* target_ptr,
                          const std::vector<const T*>& source_ptrs,
                          const std::array<int, 4>& target_size,
                          const std::vector<std::array<int, 4>>& source_sizes,
                          const std::vector<T>& scale_input_to_net_inputs = {
                              1.f});

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
