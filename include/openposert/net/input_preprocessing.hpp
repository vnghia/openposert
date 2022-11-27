#pragma once

#include <opencv4/opencv2/core/hal/interface.h>

#include <cstddef>
#include <memory>
#include <opencv2/core/core.hpp>

#include "openposert/utilities/fast_math.hpp"

namespace openposert {

class InputPreprocessing {
 public:
  InputPreprocessing() {}

  // input_data should point to a BGR image
  // no memory is allocated in this preprocessing
  InputPreprocessing(unsigned char* input_data, float* normalized_data,
                     std::size_t input_width, std::size_t input_height,
                     std::size_t input_channels, float* net_input_data,
                     std::size_t net_input_width, std::size_t net_input_height)
      : input_data_(input_data),
        normalized_data_(normalized_data),
        input_width_(input_width),
        input_height_(input_height),
        input_channels_(input_channels),
        net_input_data_(net_input_data),
        net_input_width_(net_input_width),
        net_input_height_(net_input_height),
        scale_factor_(fast_min(
            static_cast<float>(net_input_width_ - 1) / (input_width_ - 1),
            static_cast<float>(net_input_height_ - 1) / (input_height_ - 1))),
        input_mat_(input_height_, input_width_, CV_8UC3, input_data_),
        resized_mat_(net_input_height_, net_input_width_, CV_8UC3,
                     normalized_data_),
        net_input_mat_(net_input_height_, net_input_width_, CV_32FC3,
                       net_input_data_) {}

  void preprocessing_cpu();
  void preprocessing_gpu();

 private:
  unsigned char* input_data_;
  float* normalized_data_;

  std::size_t input_width_;
  std::size_t input_height_;
  std::size_t input_channels_;

  float* net_input_data_;

  std::size_t net_input_width_;
  std::size_t net_input_height_;

  float scale_factor_;

  cv::Mat input_mat_;
  cv::Mat resized_mat_;
  cv::Mat net_input_mat_;
};

}  // namespace openposert
