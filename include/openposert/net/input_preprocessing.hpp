#pragma once

#include <cstddef>
#include <memory>

namespace openposert {

class InputPreprocessing {
 public:
  InputPreprocessing() {}

  InputPreprocessing(unsigned char* input_data, std::size_t input_width,
                     std::size_t input_height, std::size_t input_channels,
                     float* net_input_data, std::size_t net_input_width,
                     std::size_t net_input_height);

  void preprocessing_gpu();

 private:
  unsigned char* input_data_;
  std::shared_ptr<void> normalized_data_;

  std::size_t input_width_;
  std::size_t input_height_;
  std::size_t input_channels_;

  float* net_input_data_;

  std::size_t net_input_width_;
  std::size_t net_input_height_;

  float scale_factor_;
};

}  // namespace openposert
