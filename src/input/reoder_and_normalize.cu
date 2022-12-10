#include "cuda_fp16.h"
#include "openposert/gpu/cuda.hpp"
#include "openposert/input/reorder_and_normalize.hpp"

namespace openposert {

__global__ void reorder_and_normalize_kernel(__half* target_ptr,
                                             const uint8_t* const src_ptr,
                                             const int width, const int height,
                                             const int channels) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto c = (blockIdx.z * blockDim.z) + threadIdx.z;
  if (x < width && y < height) {
    const auto origin_frame_ptr_offset_y = y * width;
    const auto channel_offset = c * width * height;
    const auto target_index = channel_offset + y * width + x;
    const auto src_index = (origin_frame_ptr_offset_y + x) * channels + c;
    target_ptr[target_index] = src_ptr[src_index] / 256.f - 0.5f;
  }
}

void reorder_and_normalize(__half* target_ptr, const uint8_t* const src_ptr,
                           const int width, const int height,
                           const int channels) {
  const dim3 threads_per_block{32, 1, 1};
  const dim3 num_blocks{get_number_cuda_blocks(width, threads_per_block.x),
                        get_number_cuda_blocks(height, threads_per_block.y),
                        get_number_cuda_blocks(channels, threads_per_block.z)};
  reorder_and_normalize_kernel<<<num_blocks, threads_per_block>>>(
      target_ptr, src_ptr, width, height, channels);
}

}  // namespace openposert
