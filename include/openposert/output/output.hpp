#pragma once

#include <array>
#include <memory>
#include <vector>

#include "openposert/utilities/pose_model.hpp"

namespace openposert {

class Output {
 public:
  Output() = default;

  Output(float* pose_keypoints_ptr, float* pose_scores_ptr, float scale_factor,
         int peak_dim, float* net_output_ptr, int net_output_width,
         int net_output_height, int net_output_channels, int max_joints,
         int max_peaks, const PoseModel& pose_model, bool maximize_positives,
         float nms_threshold, float inter_min_above_threshold,
         float inter_threshold, int min_subset_cnt, float min_subset_score);

  int process();

 private:
  // output
  float* pose_keypoints_ptr_;
  float* pose_scores_ptr_;
  float scale_factor_;
  int peak_dim_;

  // common
  float* net_output_ptr_;
  int net_output_width_;
  int net_output_height_;
  int net_output_channels_;

  int max_joints_;
  int max_peaks_;

  PoseModel pose_model_;

  std::vector<unsigned int> body_part_pairs_;
  std::vector<unsigned int> pose_map_idx_;
  std::shared_ptr<unsigned int[]> body_part_pairs_data_;
  std::shared_ptr<unsigned int[]> pose_map_idx_data_;
  int number_body_parts_;
  int number_body_part_pairs_;

  int paf_total_size_;

  // param
  bool maximize_positives_;
  float nms_threshold_;
  float default_nms_threshold_;
  float inter_min_above_threshold_;
  float inter_threshold_;
  int min_subset_cnt_;
  float min_subset_score_;

  // nms
  std::shared_ptr<float[]> peaks_data_;
  std::shared_ptr<int[]> kernel_data_;
  std::array<int, 4> nms_source_size_;
  std::array<int, 4> nms_target_size_;
  float nms_offset_x_;
  float nms_offset_y_;

  // paf_score
  std::shared_ptr<float[]> pair_scores_data_;

  // paf_ptr_into_vector
  int pair_connections_count_;
  std::shared_ptr<int[]> paf_sorted_index_;
  std::shared_ptr<float[]> paf_total_score_data_;
  std::shared_ptr<float[]> paf_score_data_;
  std::shared_ptr<int[]> paf_pair_index_data_;
  std::shared_ptr<int[]> paf_index_a_data_;
  std::shared_ptr<int[]> paf_index_b_data_;

  // paf_vector_into_people_vector
  int people_vector_count_;
  int person_assigned_size_;
  int people_vector_body_size_;
  std::shared_ptr<int[]> people_vector_body_data_;
  std::shared_ptr<float[]> people_vector_score_data_;
  std::shared_ptr<int[]> person_assigned_data_;
  std::shared_ptr<int[]> person_removed_data_;

  // remove_people_below_thresholds_and_fill_faces
  int number_people_;
  std::shared_ptr<int[]> valid_subset_indexes_data_;
};

}  // namespace openposert
