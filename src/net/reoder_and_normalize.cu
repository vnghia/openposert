#include <stdexcept>

#include "cuda.h"
#include "cuda_runtime.h"
#include "openposert/gpu/cuda.hpp"
#include "openposert/gpu/cuda_fast_math.hpp"
#include "openposert/net/reorder_and_normalize.hpp"

namespace openposert {

template <typename t>
__global__ void reorder_and_normalize_kernel(t* target_ptr,
                                             const unsigned char* const src_ptr,
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
    target_ptr[target_index] = t(src_ptr[src_index]) * t(1 / 256.f) - t(0.5f);
  }
}

template <typename t>
void reorder_and_normalize(t* target_ptr, const unsigned char* const src_ptr,
                           const int width, const int height,
                           const int channels) {
  const dim3 threads_per_block{32, 1, 1};
  const dim3 num_blocks{get_number_cuda_blocks(width, threads_per_block.x),
                        get_number_cuda_blocks(height, threads_per_block.y),
                        get_number_cuda_blocks(channels, threads_per_block.z)};
  reorder_and_normalize_kernel<<<num_blocks, threads_per_block>>>(
      target_ptr, src_ptr, width, height, channels);
}

template void reorder_and_normalize(float* target_ptr,
                                    const unsigned char* const src_ptr,
                                    const int width, const int height,
                                    const int channels);
template void reorder_and_normalize(double* target_ptr,
                                    const unsigned char* const src_ptr,
                                    const int width, const int height,
                                    const int channels);

}  // namespace openposert
