#include "openposert/gpu/cuda.hpp"
#include "openposert/net/input_preprocessing.hpp"
#include "openposert/net/reorder_and_normalize.hpp"
#include "openposert/net/resize_and_merge.hpp"

namespace openposert {

void InputPreprocessing::preprocessing_gpu() {
  reorder_and_normalize(normalized_data_, input_data_, input_width_,
                        input_height_, input_channels_);
  resize_and_pad_rbg_gpu(net_input_data_, normalized_data_, input_width_,
                         input_height_, net_input_width_, net_input_height_,
                         scale_factor_);
}

}  // namespace openposert
