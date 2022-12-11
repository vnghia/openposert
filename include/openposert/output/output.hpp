#pragma once

#include <array>
#include <memory>
#include <vector>

#include "cuda_fp16.h"
#include "half.hpp"
#include "openposert/utilities/pose_model.hpp"

namespace openposert {

class Output {
 public:
  Output() = default;

  Output(half_float::half* pose_keypoints_ptr,
         half_float::half* pose_scores_ptr, half_float::half scale_factor,
         int peak_dim, __half* net_output_ptr, int net_output_width,
         int net_output_height, int net_output_channels, int max_joints,
         int max_peaks, const PoseModel& pose_model, bool maximize_positives,
         __half nms_threshold, __half inter_min_above_threshold,
         __half inter_threshold, int min_subset_cnt,
         half_float::half min_subset_score);

  int process();

 private:
  // output
  half_float::half* pose_keypoints_ptr_;
  half_float::half* pose_scores_ptr_;
  half_float::half scale_factor_;
  int peak_dim_;

  // common
  __half* net_output_ptr_;
  int net_output_width_;
  int net_output_height_;
  int net_output_channels_;

  int max_joints_;
  int max_peaks_;

  PoseModel pose_model_;

  std::vector<unsigned int> body_part_pairs_;
  std::vector<unsigned int> pose_map_idx_;
  std::shared_ptr<unsigned int> body_part_pairs_data_;
  std::shared_ptr<unsigned int> pose_map_idx_data_;
  int number_body_parts_;
  int number_body_part_pairs_;

  int paf_total_size_;

  // param
  bool maximize_positives_;
  __half nms_threshold_;
  __half default_nms_threshold_;
  __half inter_min_above_threshold_;
  __half inter_threshold_;
  int min_subset_cnt_;
  half_float::half min_subset_score_;

  // nms
  std::shared_ptr<__half> peaks_data_;
  std::shared_ptr<int> kernel_data_;
  std::array<int, 4> nms_source_size_;
  std::array<int, 4> nms_target_size_;
  __half nms_offset_x_;
  __half nms_offset_y_;

  // paf_score
  std::shared_ptr<__half> pair_scores_data_;

  // paf_ptr_into_vector
  int pair_connections_count_;
  std::shared_ptr<int[]> paf_sorted_index_;
  std::shared_ptr<half_float::half[]> paf_total_score_data_;
  std::shared_ptr<half_float::half[]> paf_score_data_;
  std::shared_ptr<int[]> paf_pair_index_data_;
  std::shared_ptr<int[]> paf_index_a_data_;
  std::shared_ptr<int[]> paf_index_b_data_;

  // paf_vector_into_people_vector
  int people_vector_count_;
  int person_assigned_size_;
  int people_vector_body_size_;
  std::shared_ptr<int[]> people_vector_body_data_;
  std::shared_ptr<half_float::half[]> people_vector_score_data_;
  std::shared_ptr<int[]> person_assigned_data_;
  std::shared_ptr<int[]> person_removed_data_;

  // remove_people_below_thresholds_and_fill_faces
  int number_people_;
  std::shared_ptr<int[]> valid_subset_indexes_data_;
};

}  // namespace openposert
