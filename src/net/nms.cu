#include <array>
#include <stdexcept>

#include "openposert/core/common.hpp"
#include "openposert/gpu/cuda.hpp"
#include "openposert/gpu/cuda_fast_math.hpp"
#include "openposert/net/nms.hpp"
#include "thrust/device_ptr.h"
#include "thrust/scan.h"

namespace openposert {

const auto THREADS_PER_BLOCK_1D = 16u;
const auto THREADS_PER_BLOCK = 512u;

// note: shared memory made this function slower, from 1.2 ms to about 2 ms.
template <typename t>
__global__ void nms_register_kernel(int* kernel_ptr, const t* const source_ptr,
                                    const int w, const int h,
                                    const t threshold) {
  // get pixel location (x,y)
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  const auto channel = blockIdx.z * blockDim.z + threadIdx.z;
  const auto channel_offset = channel * w * h;
  const auto index = y * w + x;

  auto* kernel_ptr_offset = &kernel_ptr[channel_offset];
  const t* const source_ptr_offset = &source_ptr[channel_offset];

  if (0 < x && x < (w - 1) && 0 < y && y < (h - 1)) {
    const auto value = source_ptr_offset[index];
    if (value > threshold) {
      const auto top_left = source_ptr_offset[(y - 1) * w + x - 1];
      const auto top = source_ptr_offset[(y - 1) * w + x];
      const auto top_right = source_ptr_offset[(y - 1) * w + x + 1];
      const auto left = source_ptr_offset[y * w + x - 1];
      const auto right = source_ptr_offset[y * w + x + 1];
      const auto bottom_left = source_ptr_offset[(y + 1) * w + x - 1];
      const auto bottom = source_ptr_offset[(y + 1) * w + x];
      const auto bottom_right = source_ptr_offset[(y + 1) * w + x + 1];

      if (value > top_left && value > top && value > top_right &&
          value > left && value > right && value > bottom_left &&
          value > bottom && value > bottom_right)
        kernel_ptr_offset[index] = 1;
      else
        kernel_ptr_offset[index] = 0;
    } else
      kernel_ptr_offset[index] = 0;
  } else if (x == 0 || x == (w - 1) || y == 0 || y == (h - 1))
    kernel_ptr_offset[index] = 0;
}

template <typename t>
__global__ void write_result_kernel(t* output, const int length,
                                    const int* const kernel_ptr,
                                    const t* const source_ptr, const int width,
                                    const int height, const int max_peaks,
                                    const t offset_x, const t offset_y,
                                    const int offset_target) {
  __shared__ int local[THREADS_PER_BLOCK + 1];  // one more
  __shared__ int kernel0;                       // offset for kernel
  const auto global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto channel = blockIdx.y * blockDim.y + threadIdx.y;
  const auto channel_offset_source = channel * width * height;
  const auto channel_offset = channel * offset_target;

  // we need to subtract the peak at pixel 0 of the current channel for all
  // values
  if (threadIdx.x == 0) kernel0 = kernel_ptr[channel_offset_source];
  __syncthreads();

  if (global_idx < length) {
    auto* output_offset = &output[channel_offset];
    const auto* const kernel_ptr_offset = &kernel_ptr[channel_offset_source];
    const auto* const source_ptr_offset = &source_ptr[channel_offset_source];
    local[threadIdx.x] = kernel_ptr_offset[global_idx] - kernel0;
    // last thread in the block but not globally last, load one more
    if (threadIdx.x == THREADS_PER_BLOCK - 1 && global_idx != length - 1)
      local[threadIdx.x + 1] = kernel_ptr_offset[global_idx + 1] - kernel0;
    __syncthreads();

    // see difference, except the globally last one
    if (global_idx != length - 1) {
      // a[global_idx] == a[global_idx + 1] means no peak
      if (local[threadIdx.x] != local[threadIdx.x + 1]) {
        const auto peak_index = local[threadIdx.x];  // 0-index
        const auto peak_loc_x = (int)(global_idx % width);
        const auto peak_loc_y = (int)(global_idx / width);

        // accurate peak location: considered neighbors
        if (peak_index < max_peaks)  // limitation
        {
          t x_acc = 0.f;
          t y_acc = 0.f;
          t score_acc = 0.f;
          const auto d_width = 3;
          const auto d_height = 3;
          for (auto dy = -d_height; dy <= d_height; dy++) {
            const auto y = peak_loc_y + dy;
            if (0 <= y && y < height)  // default height = 368
            {
              for (auto dx = -d_width; dx <= d_width; dx++) {
                const auto x = peak_loc_x + dx;
                if (0 <= x && x < width)  // default width = 656
                {
                  const auto score = source_ptr_offset[y * width + x];
                  if (score > 0) {
                    x_acc += x * score;
                    y_acc += y * score;
                    score_acc += score;
                  }
                }
              }
            }
          }

          // offset to keep matlab format (empirically higher acc)
          // best results for 1 scale: x + 0, y + 0.5
          // +0.5 to both to keep matlab format
          const auto output_index = (peak_index + 1) * 3;
          output_offset[output_index] = x_acc / score_acc + offset_x;
          output_offset[output_index + 1] = y_acc / score_acc + offset_y;
          output_offset[output_index + 2] =
              source_ptr_offset[peak_loc_y * width + peak_loc_x];
        }
      }
    }
    // if index 0 --> assign number of peaks (truncated to the maximum possible
    // number of peaks)
    else
      output_offset[0] =
          (local[threadIdx.x] < max_peaks ? local[threadIdx.x] : max_peaks);
  }
}

template <typename t>
void nms_gpu(t* target_ptr, int* kernel_ptr, const t* const source_ptr,
             const t threshold, const std::array<int, 4>& target_size,
             const std::array<int, 4>& source_size, const t offset_x,
             const t offset_y) {
  try {
    const auto num = source_size[0];
    const auto height = source_size[2];
    const auto width = source_size[3];
    const auto channels = target_size[1];
    const auto max_peaks = target_size[2] - 1;
    const auto image_offset = height * width;
    const auto offset_target = (max_peaks + 1) * target_size[3];

    const dim3 threads_per_block2d{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D};
    const dim3 num_blocks2d{
        get_number_cuda_blocks(width, threads_per_block2d.x),
        get_number_cuda_blocks(height, threads_per_block2d.y)};
    const dim3 threads_per_block1d{THREADS_PER_BLOCK};
    const dim3 num_blocks1d{
        get_number_cuda_blocks(image_offset, threads_per_block1d.x)};

    const dim3 threads_per_block_register{THREADS_PER_BLOCK_1D,
                                          THREADS_PER_BLOCK_1D, 1};
    const dim3 num_blocks_register{
        get_number_cuda_blocks(width, threads_per_block_register.x),
        get_number_cuda_blocks(height, threads_per_block_register.y),
        get_number_cuda_blocks(num * channels, threads_per_block_register.z)};
    nms_register_kernel<<<num_blocks_register, threads_per_block_register>>>(
        kernel_ptr, source_ptr, width, height, threshold);

    auto kernel_thrust_ptr = thrust::device_pointer_cast(kernel_ptr);
    thrust::exclusive_scan(kernel_thrust_ptr,
                           kernel_thrust_ptr + num * channels * image_offset,
                           kernel_thrust_ptr);

    const dim3 threads_per_block_write{THREADS_PER_BLOCK, 1};
    const dim3 num_blocks_write{
        get_number_cuda_blocks(image_offset, threads_per_block_write.x),
        get_number_cuda_blocks(num * channels, threads_per_block_write.z)};
    write_result_kernel<<<num_blocks_write, threads_per_block_write>>>(
        target_ptr, image_offset, kernel_ptr, source_ptr, width, height,
        max_peaks, offset_x, offset_y, offset_target);

    cuda_check(__LINE__, __FUNCTION__, __FILE__);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template void nms_gpu(float* target_ptr, int* kernel_ptr,
                      const float* const source_ptr, const float threshold,
                      const std::array<int, 4>& target_size,
                      const std::array<int, 4>& source_size,
                      const float offset_x, const float offset_y);
template void nms_gpu(double* target_ptr, int* kernel_ptr,
                      const double* const source_ptr, const double threshold,
                      const std::array<int, 4>& target_size,
                      const std::array<int, 4>& source_size,
                      const double offset_x, const double offset_y);

}  // namespace openposert
