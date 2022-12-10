#pragma once

#include <cstdint>

#include "cuda_fp16.h"

namespace openposert {

void reorder_and_normalize(__half* target_ptr, const uint8_t* const src_ptr,
                           const int width, const int height,
                           const int channels);

}  // namespace openposert
