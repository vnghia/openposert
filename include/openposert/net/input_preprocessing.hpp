#pragma once

#include <cstddef>
#include <memory>

namespace openposert {

using std::size_t;

class InputPreprocessing {
 public:
  InputPreprocessing() {}

  InputPreprocessing(unsigned char* input_data, size_t input_width,
                     size_t input_height, size_t input_channels,
                     float* net_input_data, size_t net_input_width,
                     size_t net_input_height);

  void preprocessing_gpu();

 private:
  unsigned char* input_data_;
  std::shared_ptr<void> normalized_data_;

  size_t input_width_;
  size_t input_height_;
  size_t input_channels_;

  float* net_input_data_;

  size_t net_input_width_;
  size_t net_input_height_;

  float scale_factor_;
};

}  // namespace openposert
