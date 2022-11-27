#include "openposert/net/input_preprocessing.hpp"

#include "openposert/utilities/opencv.hpp"

namespace openposert {

void InputPreprocessing::preprocessing_cpu() {
  resize_fixed_aspect_ratio(resized_mat_, input_mat_, scale_factor_,
                            net_input_width_, net_input_height_);
  u_char_cv_mat_to_float_ptr(net_input_data_, resized_mat_, 1);
}

}  // namespace openposert
