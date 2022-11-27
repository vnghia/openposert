#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace openposert {

void u_char_cv_mat_to_float_ptr(float* float_ptr_image,
                                const cv::Mat& mat_image, const int normalize);

void resize_fixed_aspect_ratio(cv::Mat& resized_input, const cv::Mat& input,
                               const double scale_factor, int target_width,
                               int target_height,
                               const int border_mode = cv::BORDER_CONSTANT,
                               const cv::Scalar& border_value = cv::Scalar{0, 0,
                                                                           0});

}  // namespace openposert
