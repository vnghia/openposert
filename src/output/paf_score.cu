#include <algorithm>

#include "cuda_runtime.h"
#include "openposert/output/paf_score.hpp"

namespace openposert {

__device__ float get_score_ab(const float* body_part_a,
                              const float* body_part_b, const float* map_x,
                              const float* map_y, const int heatmap_width,
                              const int heatmap_height,
                              const float inter_threshold,
                              const float inter_min_above_threshold,
                              const float default_nms_threshold) {
  const auto vector_a_to_bx = body_part_b[0] - body_part_a[0];
  const auto vector_a_to_by = body_part_b[1] - body_part_a[1];
  const auto vector_a_to_b_max =
      std::max(std::abs(vector_a_to_bx), std::abs(vector_a_to_by));
  const auto number_points_in_line = std::max(
      5, std::min(25, static_cast<int>(std::round(std::sqrt(
                          static_cast<float>(5) * vector_a_to_b_max)))));
  const auto vector_norm = std::sqrt(vector_a_to_bx * vector_a_to_bx +
                                     vector_a_to_by * vector_a_to_by);

  // if the peaks_ptr are coincident. don'float connect them.
  if (vector_norm > static_cast<float>(1e-6)) {
    const auto s_x = body_part_a[0];
    const auto s_y = body_part_a[1];
    const auto vector_a_to_b_norm_x = vector_a_to_bx / vector_norm;
    const auto vector_a_to_b_norm_y = vector_a_to_by / vector_norm;

    auto sum = static_cast<float>(0.f);
    auto count = 0;
    const auto number_points_in_line_half =
        static_cast<float>(number_points_in_line);
    const auto vector_a_to_bx_in_line =
        vector_a_to_bx / number_points_in_line_half;
    const auto vector_a_to_by_in_line =
        vector_a_to_by / number_points_in_line_half;
    for (auto lm = 0; lm < number_points_in_line; lm++) {
      const auto m_x =
          std::min(heatmap_width - 1,
                   static_cast<int>(std::round(
                       s_x + static_cast<float>(lm) * vector_a_to_bx_in_line)));
      const auto m_y =
          std::min(heatmap_height - 1,
                   static_cast<int>(std::round(
                       s_y + static_cast<float>(lm) * vector_a_to_by_in_line)));
      const auto idx = m_y * heatmap_width + m_x;
      const auto score = (vector_a_to_b_norm_x * map_x[idx] +
                          vector_a_to_b_norm_y * map_y[idx]);
      if (score > inter_threshold) {
        sum += score;
        count++;
      }
    }
    // return paf score
    const auto count_half = static_cast<float>(count);
    if (count_half / number_points_in_line_half > inter_min_above_threshold)
      return sum / count_half;
    else {
      // ideally, if distance_ab = 0, paf is 0 between a and b, provoking a
      // false negative to fix it, we consider paf-connected keypoints very
      // close to have a minimum paf score, such that:
      //     1. it will consider very close keypoints (where the paf is 0)
      //     2. but it will not automatically connect them (case paf score = 1),
      //     or real paf might got
      //        missing
      const auto l2_dist = std::sqrt(vector_a_to_bx * vector_a_to_bx +
                                     vector_a_to_by * vector_a_to_by);
      const auto threshold =
          std::sqrt(heatmap_width * heatmap_height) /
          static_cast<float>(150);  // 3.3 for 368x656, 6.6 for 2x resolution
      if (l2_dist < threshold)
        return default_nms_threshold +
               static_cast<float>(1e-6);  // without 1e-6 will not work
                                          // because i use strict greater
    }
  }
  return -1;
}

__global__ void paf_score_kernel(
    float* pair_scores_ptr, const float* const heat_map_ptr,
    const float* const peaks_ptr, const unsigned int* const body_part_pairs_ptr,
    const unsigned int* const map_idx_ptr, const unsigned int max_peaks,
    const int number_body_part_pairs, const int heatmap_width,
    const int heatmap_height, const float inter_threshold,
    const float inter_min_above_threshold, const float default_nms_threshold) {
  const auto peak_b = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto peak_a = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto pair_index = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (peak_a < max_peaks && peak_b < max_peaks) {
    const auto base_index = 2 * pair_index;
    const auto part_a = body_part_pairs_ptr[base_index];
    const auto part_b = body_part_pairs_ptr[base_index + 1];

    const float number_peaks_a = peaks_ptr[3 * part_a * (max_peaks + 1)];
    const float number_peaks_b = peaks_ptr[3 * part_b * (max_peaks + 1)];

    const auto output_index =
        (pair_index * max_peaks + peak_a) * max_peaks + peak_b;
    if (static_cast<float>(peak_a) < number_peaks_a &&
        static_cast<float>(peak_b) < number_peaks_b) {
      const auto map_idx_x = map_idx_ptr[base_index];
      const auto map_idx_y = map_idx_ptr[base_index + 1];

      const float* const body_part_a =
          peaks_ptr + (3 * (part_a * (max_peaks + 1) + peak_a + 1));
      const float* const body_part_b =
          peaks_ptr + (3 * (part_b * (max_peaks + 1) + peak_b + 1));
      const float* const map_x =
          heat_map_ptr + map_idx_x * heatmap_width * heatmap_height;
      const float* const map_y =
          heat_map_ptr + map_idx_y * heatmap_width * heatmap_height;
      pair_scores_ptr[output_index] = get_score_ab(
          body_part_a, body_part_b, map_x, map_y, heatmap_width, heatmap_height,
          inter_threshold, inter_min_above_threshold, default_nms_threshold);
    } else
      pair_scores_ptr[output_index] = -1;
  }
}

}  // namespace openposert
