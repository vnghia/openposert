#include "openposert/gpu/cuda.hpp"
#include "openposert/gpu/cuda_fast_math.hpp"
#include "openposert/net/resize_and_merge.hpp"

namespace openposert {

const auto THREADS_PER_BLOCK_1D = 16u;

template <typename T>
__global__ void fill_kernel(T* target_ptr, const T* const source_ptr,
                            const int n) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (x < n) target_ptr[x] = source_ptr[x];
}

template <typename T>
__global__ void resize_kernel(T* target_ptr, const T* const source_ptr,
                              const int width_source, const int height_source,
                              const int width_target, const int height_target) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;
  if (x < width_target && y < height_target) {
    const auto source_area = width_source * height_source;
    const auto target_area = width_target * height_target;
    const T x_source = (x + T(0.5f)) * width_source / T(width_target) - T(0.5f);
    const T y_source =
        (y + T(0.5f)) * height_source / T(height_target) - T(0.5f);
    const T* const source_ptr_channel = source_ptr + channel * source_area;
    target_ptr[channel * target_area + y * width_target + x] =
        bicubic_interpolate(source_ptr_channel, x_source, y_source,
                            width_source, height_source, width_source);
  }
}

template <typename T>
__global__ void resize_and_pad_kernel(T* target_ptr, const T* const source_ptr,
                                      const int width_source,
                                      const int height_source,
                                      const int width_target,
                                      const int height_target,
                                      const T rescale_factor) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;
  if (x < width_target && y < height_target) {
    const auto target_area = width_target * height_target;
    if (x < width_source * rescale_factor &&
        y < height_source * rescale_factor) {
      const auto source_area = width_source * height_source;
      const T x_source = (x + T(0.5f)) / T(rescale_factor) - T(0.5f);
      const T y_source = (y + T(0.5f)) / T(rescale_factor) - T(0.5f);
      const T* const source_ptr_channel = source_ptr + channel * source_area;
      target_ptr[channel * target_area + y * width_target + x] =
          bicubic_interpolate(source_ptr_channel, x_source, y_source,
                              width_source, height_source, width_source);
    } else
      target_ptr[channel * target_area + y * width_target + x] = 0;
  }
}

template <typename T>
__global__ void resize_and_pad_kernel(
    T* target_ptr, const unsigned char* const source_ptr,
    const int width_source, const int height_source, const int width_target,
    const int height_target, const T rescale_factor) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;
  if (x < width_target && y < height_target) {
    const auto target_area = width_target * height_target;
    if (x < width_source * rescale_factor &&
        y < height_source * rescale_factor) {
      const auto source_area = width_source * height_source;
      const T x_source = (x + T(0.5f)) / T(rescale_factor) - T(0.5f);
      const T y_source = (y + T(0.5f)) / T(rescale_factor) - T(0.5f);
      const unsigned char* source_ptr_channel =
          source_ptr + channel * source_area;
      target_ptr[channel * target_area + y * width_target + x] =
          bicubic_interpolate(source_ptr_channel, x_source, y_source,
                              width_source, height_source, width_source);
    } else
      target_ptr[channel * target_area + y * width_target + x] = 0;
  }
}

template <typename T>
void resize_and_pad_rbg_gpu(T* target_ptr, const T* const src_ptr,
                            const int width_source, const int height_source,
                            const int width_target, const int height_target,
                            const T scale_factor) {
  const auto channels = 3;
  const dim3 threads_per_block{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D, 1};
  const dim3 num_blocks{
      get_number_cuda_blocks(width_target, threads_per_block.x),
      get_number_cuda_blocks(height_target, threads_per_block.y),
      get_number_cuda_blocks(channels, threads_per_block.z)};
  resize_and_pad_kernel<<<num_blocks, threads_per_block>>>(
      target_ptr, src_ptr, width_source, height_source, width_target,
      height_target, scale_factor);
}

template <typename T>
void resize_and_pad_rbg_gpu(T* target_ptr, const unsigned char* const src_ptr,
                            const int width_source, const int height_source,
                            const int width_target, const int height_target,
                            const T scale_factor)

{
  const auto channels = 3;
  const dim3 threads_per_block{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D, 1};
  const dim3 num_blocks{
      get_number_cuda_blocks(width_target, threads_per_block.x),
      get_number_cuda_blocks(height_target, threads_per_block.y),
      get_number_cuda_blocks(channels, threads_per_block.z)};
  resize_and_pad_kernel<<<num_blocks, threads_per_block>>>(
      target_ptr, src_ptr, width_source, height_source, width_target,
      height_target, scale_factor);
}

template void resize_and_pad_rbg_gpu(
    float* target_ptr, const float* const src_ptr, const int width_source,
    const int height_source, const int width_target, const int height_target,
    const float scale_factor);
template void resize_and_pad_rbg_gpu(
    double* target_ptr, const double* const src_ptr, const int width_source,
    const int height_source, const int width_target, const int height_target,
    const double scale_factor);

template void resize_and_pad_rbg_gpu(
    float* target_ptr, const unsigned char* const src_ptr,
    const int width_source, const int height_source, const int width_target,
    const int height_target, const float scale_factor);
template void resize_and_pad_rbg_gpu(
    double* target_ptr, const unsigned char* const src_ptr,
    const int width_source, const int height_source, const int width_target,
    const int height_target, const double scale_factor);

}  // namespace openposert
