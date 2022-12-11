#include "cuda_fp16.h"
#include "openposert/input/resize_and_pad_rbg.hpp"
#include "openposert/utilities/cuda.hpp"
#include "openposert/utilities/interpolate.hpp"

namespace openposert {

__global__ void resize_and_pad_kernel(
    __half* target_ptr, const __half* const source_ptr, const int width_source,
    const int height_source, const int width_target, const int height_target,
    const float rescale_factor) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;
  if (x < width_target && y < height_target) {
    const auto target_area = width_target * height_target;
    if (x < width_source * rescale_factor &&
        y < height_source * rescale_factor) {
      const auto source_area = width_source * height_source;
      const __half x_source =
          (static_cast<__half>(x) + static_cast<__half>(0.5f)) /
              static_cast<__half>(rescale_factor) -
          static_cast<__half>(0.5f);
      const __half y_source =
          (static_cast<__half>(y) + static_cast<__half>(0.5f)) /
              static_cast<__half>(rescale_factor) -
          static_cast<__half>(0.5f);
      const __half* const source_ptr_channel =
          source_ptr + channel * source_area;
      target_ptr[channel * target_area + y * width_target + x] =
          bicubic_interpolate(source_ptr_channel, x_source, y_source,
                              width_source, height_source, width_source);
    } else
      target_ptr[channel * target_area + y * width_target + x] = 0;
  }
}

void resize_and_pad_rbg(__half* target_ptr, const __half* const src_ptr,
                        const int width_source, const int height_source,
                        const int width_target, const int height_target,
                        const float scale_factor) {
  const auto channels = 3;
  const dim3 threads_per_block{16, 16, 1};
  const dim3 num_blocks{
      get_number_cuda_blocks(width_target, threads_per_block.x),
      get_number_cuda_blocks(height_target, threads_per_block.y),
      get_number_cuda_blocks(channels, threads_per_block.z)};
  resize_and_pad_kernel<<<num_blocks, threads_per_block>>>(
      target_ptr, src_ptr, width_source, height_source, width_target,
      height_target, scale_factor);
}

}  // namespace openposert
