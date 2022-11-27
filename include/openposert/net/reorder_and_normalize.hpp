#pragma once

namespace openposert {

template <typename T>
void reorder_and_normalize(T* target_ptr, const unsigned char* const src_ptr,
                           const int width, const int height,
                           const int channels);

}  // namespace openposert
