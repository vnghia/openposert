#include "minrt/utils.hpp"
#include "openposert/gpu/cuda.hpp"
#include "openposert/net/input_preprocessing.hpp"
#include "openposert/net/reorder_and_normalize.hpp"
#include "openposert/net/resize_and_merge.hpp"
#include "openposert/utilities/fast_math.hpp"

namespace openposert {

using namespace minrt;

InputPreprocessing::InputPreprocessing(
    unsigned char* input_data, std::size_t input_width,
    std::size_t input_height, std::size_t input_channels, float* net_input_data,
    std::size_t net_input_width, std::size_t net_input_height)
    : input_data_(input_data),
      input_width_(input_width),
      input_height_(input_height),
      input_channels_(input_channels),
      net_input_data_(net_input_data),
      net_input_width_(net_input_width),
      net_input_height_(net_input_height),
      scale_factor_(fast_min(
          static_cast<float>(net_input_width_ - 1) / (input_width_ - 1),
          static_cast<float>(net_input_height_ - 1) / (input_height_ - 1))) {
  auto normalized_input_size =
      input_width * input_height * input_channels * sizeof(float);
  spdlog::info("allocated {} byte for normalized input data",
               normalized_input_size);
  normalized_data_ = cuda_malloc(normalized_input_size);
}

void InputPreprocessing::preprocessing_gpu() {
  reorder_and_normalize(static_cast<float*>(normalized_data_.get()),
                        input_data_, input_width_, input_height_,
                        input_channels_);
  resize_and_pad_rbg_gpu(net_input_data_,
                         static_cast<float*>(normalized_data_.get()),
                         input_width_, input_height_, net_input_width_,
                         net_input_height_, scale_factor_);
}

}  // namespace openposert
