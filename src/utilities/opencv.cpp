#include "openposert/utilities/opencv.hpp"

#include <stdexcept>

#include "immintrin.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "openposert/core/common.hpp"

namespace openposert {

void u_char_cv_mat_to_float_ptr(float* ptr_image, const cv::Mat& mat_image,
                                const int normalize) {
  const int width = mat_image.cols;
  const int height = mat_image.rows;
  const int channels = mat_image.channels();

  const auto* const origin_frame_ptr =
      mat_image.data;  // cv::mat.data is always uchar
  for (auto c = 0; c < channels; c++) {
    const auto ptr_image_offset_c = c * height;
    for (auto y = 0; y < height; y++) {
      const auto ptr_image_offset_y = (ptr_image_offset_c + y) * width;
      const auto origin_frame_ptr_offset_y = y * width;
      for (auto x = 0; x < width; x++)
        ptr_image[ptr_image_offset_y + x] = float(
            origin_frame_ptr[(origin_frame_ptr_offset_y + x) * channels + c]);
    }
  }

  if (normalize == 1) {
    cv::Mat ptr_image_wrapper(
        height * width * 3, 1, CV_32FC1,
        ptr_image);  // CV_32FC3 warns about
                     // https://github.com/opencv/opencv/issues/16739
    ptr_image_wrapper = ptr_image_wrapper * (1 / 256.f) - 0.5f;
  }
}

void resize_fixed_aspect_ratio(cv::Mat& resized_input, const cv::Mat& input,
                               const double scale_factor, int target_width,
                               int target_height, const int border_mode,
                               const cv::Scalar& border_value) {
  try {
    const cv::Size target_size{target_width, target_height};
    cv::Mat M = cv::Mat::eye(2, 3, CV_64F);
    M.at<double>(0, 0) = scale_factor;
    M.at<double>(1, 1) = scale_factor;
    if (scale_factor != 1. || target_size != input.size())
      cv::warpAffine(input, resized_input, M, target_size,
                     (scale_factor > 1. ? cv::INTER_CUBIC : cv::INTER_AREA),
                     border_mode, border_value);
    else
      input.copyTo(resized_input);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

}  // namespace openposert
