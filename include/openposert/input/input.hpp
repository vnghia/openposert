#pragma once

#include <cstdint>
#include <memory>

#include "cuda_fp16.h"

namespace openposert {

class Input {
 public:
  Input() = default;

  Input(uint8_t* input_ptr, int input_width, int input_height,
        int input_channels, __half* net_input_ptr, int net_input_width,
        int net_input_height);

  void process();

 private:
  uint8_t* input_ptr_;
  std::shared_ptr<__half> normalized_data_;

  int input_width_;
  int input_height_;
  int input_channels_;

  __half* net_input_ptr_;

  int net_input_width_;
  int net_input_height_;

  __half scale_factor_;
};

}  // namespace openposert
