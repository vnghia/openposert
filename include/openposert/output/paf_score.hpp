#pragma once

#include <algorithm>

#include "cuda_fp16.h"
#include "cuda_runtime.h"

namespace openposert {

__device__ __half get_score_ab(const __half* body_part_a,
                               const __half* body_part_b, const __half* map_x,
                               const __half* map_y, const int heatmap_width,
                               const int heatmap_height,
                               const __half inter_threshold,
                               const __half inter_min_above_threshold,
                               const __half default_nms_threshold);

__global__ void paf_score_kernel(
    __half* pair_scores_ptr, const __half* const heat_map_ptr,
    const __half* const peaks_ptr,
    const unsigned int* const body_part_pairs_ptr,
    const unsigned int* const map_idx_ptr, const unsigned int max_peaks,
    const int number_body_part_pairs, const int heatmap_width,
    const int heatmap_height, const __half inter_threshold,
    const __half inter_min_above_threshold, const __half default_nms_threshold);

}  // namespace openposert
