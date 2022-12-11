#pragma once

#include <algorithm>

#include "cuda_runtime.h"

namespace openposert {

__device__ float get_score_ab(const float* body_part_a,
                              const float* body_part_b, const float* map_x,
                              const float* map_y, const int heatmap_width,
                              const int heatmap_height,
                              const float inter_threshold,
                              const float inter_min_above_threshold,
                              const float default_nms_threshold);

__global__ void paf_score_kernel(
    float* pair_scores_ptr, const float* const heat_map_ptr,
    const float* const peaks_ptr, const unsigned int* const body_part_pairs_ptr,
    const unsigned int* const map_idx_ptr, const unsigned int max_peaks,
    const int number_body_part_pairs, const int heatmap_width,
    const int heatmap_height, const float inter_threshold,
    const float inter_min_above_threshold, const float default_nms_threshold);

}  // namespace openposert
