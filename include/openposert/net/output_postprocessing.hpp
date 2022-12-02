#pragma once

#include <memory>

#include "openposert/pose/enum.hpp"

namespace openposert {

class OutputPostprocessing {
 public:
  OutputPostprocessing() {}

  OutputPostprocessing(float* pose_keypoints, float* pose_scores,
                       float* net_output_ptr, int net_output_width,
                       int net_output_height, int peak_dim, float scale_factor,
                       int number_body_parts, int number_body_part_pairs,
                       int max_peaks, int min_subset_cnt,
                       float min_subset_score, bool maximize_positives,
                       float inter_threshold, float inter_min_above_threshold,
                       float default_nms_threshold, float* peaks_ptr,
                       float* pair_scores_ptr,
                       unsigned int* body_part_pairs_ptr,
                       unsigned int* pose_map_idx_ptr);

  int postprocessing_gpu();

  float* net_output_ptr;
  int net_output_height;
  int net_output_width;

  int peak_dim;

  float scale_factor;

  int number_body_parts;
  int number_body_part_pairs;
  int max_peaks;

  int min_subset_cnt;
  float min_subset_score;
  bool maximize_positives;
  float inter_threshold;
  float inter_min_above_threshold;
  float default_nms_threshold;

  float* peaks_ptr;
  float* pair_scores_ptr;
  unsigned int* body_part_pairs_ptr;
  unsigned int* pose_map_idx_ptr;

  int paf_total_size;

 private:
  void paf_ptr_into_vector_gpu(
      int* sorted_ptr, float* total_score_ptr, float* paf_score_ptr,
      int* pair_index_ptr, int* index_a_ptr, int* index_b_ptr,
      const int total_size, const float* const pair_scores,
      const float* const peaks_ptr, const int max_peaks,
      const unsigned int* body_part_pairs,
      const unsigned int number_body_part_pairs, int& pair_connections_count);

  void paf_vector_into_people_vector_gpu(
      int* people_vector_body_ptr, float* people_vector_score_ptr,
      int* person_assigned_ptr, int* person_removed_ptr,
      const int* const paf_sorted_ptr, const float* const paf_score_ptr,
      const int* const paf_pair_index_ptr, const int* const paf_index_a_ptr,
      const int* const paf_index_b_ptr, const int pair_connections_count,
      const float* const peaks_ptr, const int max_peaks,
      const unsigned int* body_part_pairs, const unsigned int number_body_parts,
      int& people_vector_count);

  void remove_people_below_thresholds_and_fill_faces_gpu(
      int* valid_subset_indexes_ptr, int& number_people,
      int* people_vector_body_ptr, float* people_vector_score_ptr,
      int* person_removed_ptr, const int people_vector_count,
      const unsigned int number_body_parts, const int min_subset_cnt,
      const float min_subset_score, const bool maximize_positives,
      const float* const peaks_ptr);

  void people_vector_to_people_array_gpu(
      float* pose_keypoints, float* pose_scores, const float scale_factor,
      const int* const people_vector_body_ptr,
      const float* const people_vector_score_ptr,
      const int* const valid_subset_indexes_ptr, const int people_vector_count,
      const float* const peaks_ptr, const int number_people,
      const unsigned int number_body_parts,
      const unsigned int number_body_part_pairs);

  PoseModel pose_model;

  std::shared_ptr<void> paf_sorted_index_;
  std::shared_ptr<void> paf_total_score_data_;
  std::shared_ptr<void> paf_score_data_;
  std::shared_ptr<void> paf_pair_index_data_;
  std::shared_ptr<void> paf_index_a_data_;
  std::shared_ptr<void> paf_index_b_data_;

  int pair_connections_count_;

  std::shared_ptr<void> people_vector_body_data_;
  std::shared_ptr<void> people_vector_score_data_;
  std::shared_ptr<void> person_assigned_data_;
  std::shared_ptr<void> person_removed_data_;

  int people_vector_count_;

  std::shared_ptr<void> valid_subset_indexes_data_;

  float* pose_keypoints_;
  float* pose_scores_;

  int number_people_;
};

}  // namespace openposert
