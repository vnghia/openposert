#include "openposert/core/common.hpp"
#include "openposert/gpu/cuda.hpp"
#include "openposert/gpu/cuda_fast_math.hpp"
#include "openposert/net/resize_and_merge.hpp"

namespace openposert {

const auto THREADS_PER_BLOCK = 256u;
const auto THREADS_PER_BLOCK_1D = 16u;

template <typename t>
__global__ void fill_kernel(t* target_ptr, const t* const source_ptr,
                            const int n) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (x < n) target_ptr[x] = source_ptr[x];
}

template <typename t>
__global__ void resize_kernel(t* target_ptr, const t* const source_ptr,
                              const int width_source, const int height_source,
                              const int width_target, const int height_target) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;
  if (x < width_target && y < height_target) {
    const auto source_area = width_source * height_source;
    const auto target_area = width_target * height_target;
    const t x_source = (x + t(0.5f)) * width_source / t(width_target) - t(0.5f);
    const t y_source =
        (y + t(0.5f)) * height_source / t(height_target) - t(0.5f);
    const t* const source_ptr_channel = source_ptr + channel * source_area;
    target_ptr[channel * target_area + y * width_target + x] =
        bicubic_interpolate(source_ptr_channel, x_source, y_source,
                            width_source, height_source, width_source);
  }
}

template <typename t>
__global__ void resize_and_pad_kernel(t* target_ptr, const t* const source_ptr,
                                      const int width_source,
                                      const int height_source,
                                      const int width_target,
                                      const int height_target,
                                      const t rescale_factor) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;
  if (x < width_target && y < height_target) {
    const auto target_area = width_target * height_target;
    if (x < width_source * rescale_factor &&
        y < height_source * rescale_factor) {
      const auto source_area = width_source * height_source;
      const t x_source = (x + t(0.5f)) / t(rescale_factor) - t(0.5f);
      const t y_source = (y + t(0.5f)) / t(rescale_factor) - t(0.5f);
      const t* const source_ptr_channel = source_ptr + channel * source_area;
      target_ptr[channel * target_area + y * width_target + x] =
          bicubic_interpolate(source_ptr_channel, x_source, y_source,
                              width_source, height_source, width_source);
    } else
      target_ptr[channel * target_area + y * width_target + x] = 0;
  }
}

template <typename t>
__global__ void resize_and_pad_kernel(
    t* target_ptr, const unsigned char* const source_ptr,
    const int width_source, const int height_source, const int width_target,
    const int height_target, const t rescale_factor) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;
  if (x < width_target && y < height_target) {
    const auto target_area = width_target * height_target;
    if (x < width_source * rescale_factor &&
        y < height_source * rescale_factor) {
      const auto source_area = width_source * height_source;
      const t x_source = (x + t(0.5f)) / t(rescale_factor) - t(0.5f);
      const t y_source = (y + t(0.5f)) / t(rescale_factor) - t(0.5f);
      const unsigned char* source_ptr_channel =
          source_ptr + channel * source_area;
      target_ptr[channel * target_area + y * width_target + x] =
          bicubic_interpolate(source_ptr_channel, x_source, y_source,
                              width_source, height_source, width_source);
    } else
      target_ptr[channel * target_area + y * width_target + x] = 0;
  }
}

template <typename t>
__global__ void resize8_times_kernel(t* target_ptr, const t* const source_ptr,
                                     const int width_source,
                                     const int height_source,
                                     const int width_target,
                                     const int height_target,
                                     const unsigned int rescale_factor) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (x < width_target && y < height_target) {
    // normal resize
    // note: the first blockIdx of each dimension behaves differently, so
    // applying old version in those
    if (blockIdx.x < 1 || blockIdx.y < 1)
    // actually it is only required for the first 4, but then i would have not
    // loaded the shared memory if ((blockIdx.x < 1 || blockIdx.y < 1) &&
    // (threadIdx.x < 4 || threadIdx.y < 4))
    {
      const auto source_area = width_source * height_source;
      const auto target_area = width_target * height_target;
      const t x_source = (x + t(0.5f)) / t(rescale_factor) - t(0.5f);
      const t y_source = (y + t(0.5f)) / t(rescale_factor) - t(0.5f);
      const t* const source_ptr_channel = source_ptr + channel * source_area;
      target_ptr[channel * target_area + y * width_target + x] =
          bicubic_interpolate(source_ptr_channel, x_source, y_source,
                              width_source, height_source, width_source);
      return;
    }

    // load shared memory
    // if resize >= 5, then #threads per block >= # elements of shared memory
    const auto shared_size = 25;  // (4+1)^2
    __shared__ t source_ptr_shared[shared_size];
    const auto shared_load_id = threadIdx.x + rescale_factor * threadIdx.y;
    if (shared_load_id < shared_size) {
      // idea: find minimum possible x and y
      const auto min_target_x = blockIdx.x * rescale_factor;
      const auto min_source_x_float =
          (min_target_x + t(0.5f)) / t(rescale_factor) - t(0.5f);
      const auto min_source_x_int = int(floor(min_source_x_float)) - 1;
      const auto min_target_y = blockIdx.y * rescale_factor;
      const auto min_source_y_float =
          (min_target_y + t(0.5f)) / t(rescale_factor) - t(0.5f);
      const auto min_source_y_int = int(floor(min_source_y_float)) - 1;
      // get current x and y
      const auto x_clean = fast_truncate_cuda(
          min_source_x_int + int(shared_load_id % 5), 0, width_source - 1);
      const auto y_clean = fast_truncate_cuda(
          min_source_y_int + int(shared_load_id / 5), 0, height_source - 1);
      // load into shared memory
      const auto source_index =
          (channel * height_source + y_clean) * width_source + x_clean;
      source_ptr_shared[shared_load_id] = source_ptr[source_index];
    }
    __syncthreads();

    // apply resize
    const auto target_area = width_target * height_target;
    const t x_source = (x + t(0.5f)) / t(rescale_factor) - t(0.5f);
    const t y_source = (y + t(0.5f)) / t(rescale_factor) - t(0.5f);
    target_ptr[channel * target_area + y * width_target + x] =
        bicubic_interpolate8_times(source_ptr_shared, x_source, y_source,
                                   width_source, height_source, threadIdx.x,
                                   threadIdx.y);
  }
}

template <typename t>
__global__ void resize_and_add_and_average_kernel(
    t* target_ptr, const int counter, const t* const scale_widths,
    const t* const scale_heights, const int* const width_sources,
    const int* const height_sources, const int width_target,
    const int height_target, const t* const source_ptr0,
    const t* const source_ptr1, const t* const source_ptr2,
    const t* const source_ptr3, const t* const source_ptr4,
    const t* const source_ptr5, const t* const source_ptr6,
    const t* const source_ptr7) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;
  // for each pixel
  if (x < width_target && y < height_target) {
    // local variable for higher speed
    t interpolated = t(0.f);
    // for each input source pointer
    for (auto i = 0; i < counter; ++i) {
      const auto source_area = width_sources[i] * height_sources[i];
      const t x_source = (x + t(0.5f)) / scale_widths[i] - t(0.5f);
      const t y_source = (y + t(0.5f)) / scale_heights[i] - t(0.5f);
      const t* const source_ptr = (i == 0   ? source_ptr0
                                   : i == 1 ? source_ptr1
                                   : i == 2 ? source_ptr2
                                   : i == 3 ? source_ptr3
                                   : i == 4 ? source_ptr4
                                   : i == 5 ? source_ptr5
                                   : i == 6 ? source_ptr6
                                            : source_ptr7);
      const t* const source_ptr_channel = source_ptr + channel * source_area;
      interpolated += bicubic_interpolate(source_ptr_channel, x_source,
                                          y_source, width_sources[i],
                                          height_sources[i], width_sources[i]);
    }
    // save into memory
    const auto target_area = width_target * height_target;
    target_ptr[channel * target_area + y * width_target + x] =
        interpolated / t(counter);
  }
}

template <typename t>
void resize_and_merge_gpu(t* target_ptr,
                          const std::vector<const t*>& source_ptrs,
                          const std::array<int, 4>& target_size,
                          const std::vector<std::array<int, 4>>& source_sizes,
                          const std::vector<t>& scale_input_to_net_inputs) {
  try {
    // sanity checks
    if (source_sizes.empty())
      error("source_sizes cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
    if (source_ptrs.size() != source_sizes.size() ||
        source_sizes.size() != scale_input_to_net_inputs.size())
      error(
          "size(source_ptrs) must match size(source_sizes) and "
          "size(scale_input_to_net_inputs). currently: " +
              std::to_string(source_ptrs.size()) + " vs. " +
              std::to_string(source_sizes.size()) + " vs. " +
              std::to_string(scale_input_to_net_inputs.size()) + ".",
          __LINE__, __FUNCTION__, __FILE__);

    // parameters
    const auto channels = target_size[1];
    const auto height_target = target_size[2];
    const auto width_target = target_size[3];

    const auto& source_size = source_sizes[0];
    const auto height_source = source_size[2];
    const auto width_source = source_size[3];

    // no multi-scale merging or no merging required
    if (source_sizes.size() == 1) {
      const auto num = source_size[0];
      if (target_size[0] > 1 || num == 1) {
        if (width_target / width_source == 1 &&
            height_target / height_source == 1) {
          const auto n = width_target * height_target * num * channels;
          const dim3 threads_per_block{THREADS_PER_BLOCK};
          const dim3 num_blocks{get_number_cuda_blocks(n, threads_per_block.x)};
          fill_kernel<<<num_blocks, threads_per_block>>>(target_ptr,
                                                         source_ptrs.at(0), n);
        } else {
          if (width_target / width_source != 8 ||
              height_target / height_source != 8)
            error(
                "kernel only implemented for 8x resize. notify us if this "
                "error appears.",
                __LINE__, __FUNCTION__, __FILE__);
          const auto rescale_factor =
              (unsigned int)std::ceil(height_target / (float)(height_source));
          const dim3 threads_per_block{rescale_factor, rescale_factor, 1};
          const dim3 num_blocks{
              get_number_cuda_blocks(width_target, threads_per_block.x),
              get_number_cuda_blocks(height_target, threads_per_block.y),
              get_number_cuda_blocks(num * channels, threads_per_block.z)};
          resize8_times_kernel<<<num_blocks, threads_per_block>>>(
              target_ptr, source_ptrs.at(0), width_source, height_source,
              width_target, height_target, rescale_factor);
        }
      }
      // old inefficient multi-scale merging
      else
        error("it should never reache this point. notify us otherwise.",
              __LINE__, __FUNCTION__, __FILE__);
    }
    // multi-scaling merging
    else {
      const auto scale_to_main_scale_width = width_target / t(width_source);
      const auto scale_to_main_scale_height = height_target / t(height_source);

      if (source_ptrs.size() > 8)
        error(
            "more than 8 scales are not implemented (yet). notify us to "
            "implement it.",
            __LINE__, __FUNCTION__, __FILE__);
      const dim3 threads_per_block{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D,
                                   1};
      const dim3 num_blocks{
          get_number_cuda_blocks(width_target, threads_per_block.x),
          get_number_cuda_blocks(height_target, threads_per_block.y),
          get_number_cuda_blocks(channels, threads_per_block.z)};
      // fill auxiliary params
      std::vector<int> width_sources_cpu(source_sizes.size());
      std::vector<int> height_sources_cpu(source_sizes.size());
      std::vector<t> scale_widths_cpu(source_sizes.size());
      std::vector<t> scale_heights_cpu(source_sizes.size());
      for (auto i = 0u; i < source_sizes.size(); ++i) {
        const auto& current_size = source_sizes.at(i);
        height_sources_cpu[i] = current_size[2];
        width_sources_cpu[i] = current_size[3];
        const auto scale_input_to_net =
            scale_input_to_net_inputs[i] / scale_input_to_net_inputs[0];
        scale_widths_cpu[i] = scale_to_main_scale_width / scale_input_to_net;
        scale_heights_cpu[i] = scale_to_main_scale_height / scale_input_to_net;
      }
      // gpu params
      int* width_sources;
      cudaMalloc((void**)&width_sources, sizeof(int) * source_sizes.size());
      cudaMemcpy(width_sources, width_sources_cpu.data(),
                 sizeof(int) * source_sizes.size(),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      int* height_sources;
      cudaMalloc((void**)&height_sources, sizeof(int) * source_sizes.size());
      cudaMemcpy(height_sources, height_sources_cpu.data(),
                 sizeof(int) * source_sizes.size(),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      t* scale_widths;
      cudaMalloc((void**)&scale_widths, sizeof(t) * source_sizes.size());
      cudaMemcpy(scale_widths, scale_widths_cpu.data(),
                 sizeof(t) * source_sizes.size(),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      t* scale_heights;
      cudaMalloc((void**)&scale_heights, sizeof(t) * source_sizes.size());
      cudaMemcpy(scale_heights, scale_heights_cpu.data(),
                 sizeof(t) * source_sizes.size(),
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      // resize each channel, add all, and get average
      resize_and_add_and_average_kernel<<<num_blocks, threads_per_block>>>(
          target_ptr, (int)source_sizes.size(), scale_widths, scale_heights,
          width_sources, height_sources, width_target, height_target,
          source_ptrs[0], source_ptrs[1], source_ptrs[2], source_ptrs[3],
          source_ptrs[4], source_ptrs[5], source_ptrs[6], source_ptrs[7]);
      // free memory
      if (width_sources != nullptr) cudaFree(width_sources);
      if (height_sources != nullptr) cudaFree(height_sources);
      if (scale_widths != nullptr) cudaFree(scale_widths);
      if (scale_heights != nullptr) cudaFree(scale_heights);
    }

    cuda_check(__LINE__, __FUNCTION__, __FILE__);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename t>
void resize_and_pad_rbg_gpu(t* target_ptr, const t* const src_ptr,
                            const int width_source, const int height_source,
                            const int width_target, const int height_target,
                            const t scale_factor) {
  try {
    const auto channels = 3;
    const dim3 threads_per_block{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D, 1};
    const dim3 num_blocks{
        get_number_cuda_blocks(width_target, threads_per_block.x),
        get_number_cuda_blocks(height_target, threads_per_block.y),
        get_number_cuda_blocks(channels, threads_per_block.z)};
    resize_and_pad_kernel<<<num_blocks, threads_per_block>>>(
        target_ptr, src_ptr, width_source, height_source, width_target,
        height_target, scale_factor);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename t>
void resize_and_pad_rbg_gpu(t* target_ptr, const unsigned char* const src_ptr,
                            const int width_source, const int height_source,
                            const int width_target, const int height_target,
                            const t scale_factor)

{
  try {
    const auto channels = 3;
    const dim3 threads_per_block{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D, 1};
    const dim3 num_blocks{
        get_number_cuda_blocks(width_target, threads_per_block.x),
        get_number_cuda_blocks(height_target, threads_per_block.y),
        get_number_cuda_blocks(channels, threads_per_block.z)};
    resize_and_pad_kernel<<<num_blocks, threads_per_block>>>(
        target_ptr, src_ptr, width_source, height_source, width_target,
        height_target, scale_factor);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template void resize_and_merge_gpu(
    float* target_ptr, const std::vector<const float*>& source_ptrs,
    const std::array<int, 4>& target_size,
    const std::vector<std::array<int, 4>>& source_sizes,
    const std::vector<float>& scale_input_to_net_inputs);
template void resize_and_merge_gpu(
    double* target_ptr, const std::vector<const double*>& source_ptrs,
    const std::array<int, 4>& target_size,
    const std::vector<std::array<int, 4>>& source_sizes,
    const std::vector<double>& scale_input_to_net_inputs);

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
