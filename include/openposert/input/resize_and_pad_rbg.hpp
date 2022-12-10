#pragma once

#include <cstdint>

#include "cuda_fp16.h"

namespace openposert {

void resize_and_pad_rbg(__half* target_ptr, const __half* const src_ptr,
                        const int source_width, const int source_height,
                        const int target_width, const int target_height,
                        const float scale_factor);

}  // namespace openposert
