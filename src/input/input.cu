#include <algorithm>
#include <cstdint>
#include <memory>

#include "cuda_fp16.h"
#include "minrt/utils.hpp"
#include "openposert/input/input.hpp"
#include "openposert/input/reorder_and_normalize.hpp"
#include "openposert/input/resize_and_pad_rbg.hpp"
#include "spdlog/spdlog.h"

namespace openposert {

using namespace minrt;

Input::Input(uint8_t* input_ptr, int input_width, int input_height,
             int input_channels, __half* net_input_ptr, int net_input_width,
             int net_input_height)
    : input_ptr_(input_ptr),
      input_width_(input_width),
      input_height_(input_height),
      input_channels_(input_channels),
      net_input_ptr_(net_input_ptr),
      net_input_width_(net_input_width),
      net_input_height_(net_input_height),
      scale_factor_(std::min(
          static_cast<float>(net_input_width_ - 1) / (input_width - 1),
          static_cast<float>(net_input_height_ - 1) / (input_height - 1))) {
  auto normalized_input_size =
      input_width * input_height * input_channels * sizeof(__half);
  spdlog::info(
      "[input] allocated {} byte for normalized input data dims=[{}, {}, {}]",
      normalized_input_size, input_channels, input_height, input_width);
  normalized_data_ = cuda_malloc<__half[]>(normalized_input_size);
}

void Input::process() {
  reorder_and_normalize(normalized_data_.get(), input_ptr_, input_width_,
                        input_height_, input_channels_);
  resize_and_pad_rbg(net_input_ptr_, normalized_data_.get(), input_width_,
                     input_height_, net_input_width_, net_input_height_,
                     scale_factor_);
}

}  // namespace openposert
