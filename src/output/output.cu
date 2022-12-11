#include <algorithm>
#include <memory>
#include <numeric>

#include "minrt/utils.hpp"
#include "openposert/output/nms.hpp"
#include "openposert/output/output.hpp"
#include "openposert/output/paf_ptr_into_vector.hpp"
#include "openposert/output/paf_score.hpp"
#include "openposert/output/paf_vector_into_people_vector.hpp"
#include "openposert/output/people_vector_to_people_array.hpp"
#include "openposert/output/remove_people_below_thresholds_and_fill_faces.hpp"
#include "openposert/utilities/cuda.hpp"

namespace openposert {

using namespace minrt;

Output::Output(float* pose_keypoints_ptr, float* pose_scores_ptr,
               float scale_factor, int peak_dim, float* net_output_ptr,
               int net_output_width, int net_output_height,
               int net_output_channels, int max_joints, int max_peaks,
               const PoseModel& pose_model, bool maximize_positives,
               float nms_threshold, float inter_min_above_threshold,
               float inter_threshold, int min_subset_cnt,
               float min_subset_score)
    : pose_keypoints_ptr_(pose_keypoints_ptr),
      pose_scores_ptr_(pose_scores_ptr),
      scale_factor_(scale_factor),
      peak_dim_(peak_dim),
      net_output_ptr_(net_output_ptr),
      net_output_width_(net_output_width),
      net_output_height_(net_output_height),
      net_output_channels_(net_output_channels),
      max_joints_(max_joints),
      max_peaks_(max_peaks),
      pose_model_(pose_model),
      body_part_pairs_(get_pose_part_pairs(pose_model_)),
      pose_map_idx_(([this]() {
        const auto number_body_part = get_pose_number_body_parts(pose_model_);
        auto pose_map_idx = get_pose_map_index(pose_model_);
        const auto offset = (add_bkg_channel(pose_model_) ? 1 : 0);
        for (auto& i : pose_map_idx) i += (number_body_part + offset);
        return pose_map_idx;
      })()),
      number_body_parts_(get_pose_number_body_parts(pose_model_)),
      number_body_part_pairs_(static_cast<int>(body_part_pairs_.size() / 2)),
      paf_total_size_(number_body_part_pairs_ * max_peaks_ * max_peaks_),
      maximize_positives_(maximize_positives),
      nms_threshold_(nms_threshold),
      default_nms_threshold_(nms_threshold),
      inter_min_above_threshold_(inter_min_above_threshold),
      inter_threshold_(inter_threshold),
      min_subset_cnt_(min_subset_cnt),
      min_subset_score_(min_subset_score),
      nms_source_size_(
          {1, net_output_channels_, net_output_height_, net_output_width_}),
      nms_target_size_({1, max_joints_, max_peaks_ + 1, peak_dim_}),
      nms_offset_x_(0.5),
      nms_offset_y_(0.5) {
  // common
  auto body_part_pair_size = body_part_pairs_.size() * sizeof(unsigned int);
  body_part_pairs_data_ =
      cuda_malloc_managed<unsigned int[]>(body_part_pair_size);
  cuda_upload(body_part_pairs_data_.get(), body_part_pairs_.data(),
              body_part_pair_size);

  auto pose_map_idx_size = pose_map_idx_.size() * sizeof(unsigned int);
  pose_map_idx_data_ = cuda_malloc_managed<unsigned int[]>(pose_map_idx_size);
  cuda_upload(pose_map_idx_data_.get(), pose_map_idx_.data(),
              pose_map_idx_size);

  // nms
  auto peaks_size = max_joints_ * (max_peaks_ + 1) * peak_dim_ * sizeof(float);
  peaks_data_ = cuda_malloc_managed<float[]>(peaks_size);
  spdlog::info("[output] allocated {} byte for peak data dims=[{}, {}, {}]",
               peaks_size, max_joints_, max_peaks_ + 1, peak_dim_);

  auto kernel_size = net_output_width_ * net_output_height_ *
                     net_output_channels_ * sizeof(int);
  kernel_data_ = cuda_malloc<int[]>(kernel_size);
  spdlog::info("[output] allocated {} byte for kernel data dims=[{}, {}, {}]",
               kernel_size, net_output_channels_, net_output_height_,
               net_output_width_);

  // paf_score
  auto pair_score_size = paf_total_size_ * sizeof(float);
  pair_scores_data_ = cuda_malloc_managed<float[]>(pair_score_size);
  spdlog::info(
      "[output] allocated {} byte for pair score data dims=[{}, {}, {}]",
      pair_score_size, number_body_part_pairs_, max_peaks_, max_peaks_);

  // paf_ptr_into_vector
  paf_sorted_index_.reset(new int[paf_total_size_]);
  paf_total_score_data_.reset(new float[paf_total_size_]);
  paf_score_data_.reset(new float[paf_total_size_]);
  paf_pair_index_data_.reset(new int[paf_total_size_]);
  paf_index_a_data_.reset(new int[paf_total_size_]);
  paf_index_b_data_.reset(new int[paf_total_size_]);

  // paf_vector_into_people_vector
  person_assigned_size_ = number_body_parts_ * max_peaks_;
  people_vector_body_size_ = paf_total_size_ * (number_body_parts_ + 1);
  people_vector_body_data_.reset(new int[people_vector_body_size_]);
  people_vector_score_data_.reset(new float[paf_total_size_]);
  person_assigned_data_.reset(new int[person_assigned_size_]);
  person_removed_data_.reset(new int[paf_total_size_]);

  // remove_people_below_thresholds_and_fill_faces
  valid_subset_indexes_data_.reset(new int[paf_total_size_]);
}

int Output::process() {
  nms(peaks_data_.get(), kernel_data_.get(), net_output_ptr_, nms_threshold_,
      nms_target_size_, nms_source_size_, nms_offset_x_, nms_offset_y_);

  const dim3 threads_per_block{128, 1, 1};
  const dim3 num_blocks{
      get_number_cuda_blocks(max_peaks_, threads_per_block.x),
      get_number_cuda_blocks(max_peaks_, threads_per_block.y),
      get_number_cuda_blocks(number_body_part_pairs_, threads_per_block.z)};
  paf_score_kernel<<<num_blocks, threads_per_block>>>(
      pair_scores_data_.get(), net_output_ptr_, peaks_data_.get(),
      body_part_pairs_data_.get(), pose_map_idx_data_.get(), max_peaks_,
      number_body_part_pairs_, net_output_width_, net_output_height_,
      inter_threshold_, inter_min_above_threshold_, default_nms_threshold_);

  pair_connections_count_ = 0;
  std::iota(paf_sorted_index_.get(), paf_sorted_index_.get() + paf_total_size_,
            0);
  paf_ptr_into_vector(paf_sorted_index_.get(), paf_total_score_data_.get(),
                      paf_score_data_.get(), paf_pair_index_data_.get(),
                      paf_index_a_data_.get(), paf_index_b_data_.get(),
                      pair_connections_count_,
                      reinterpret_cast<float*>(pair_scores_data_.get()),
                      reinterpret_cast<float*>(peaks_data_.get()), max_peaks_,
                      body_part_pairs_data_.get(), number_body_part_pairs_);

  people_vector_count_ = 0;
  std::fill_n(person_assigned_data_.get(), person_assigned_size_, -1);
  std::fill_n(people_vector_body_data_.get(), people_vector_body_size_, 0);
  paf_vector_into_people_vector(
      people_vector_body_data_.get(), people_vector_score_data_.get(),
      person_assigned_data_.get(), person_removed_data_.get(),
      people_vector_count_, paf_sorted_index_.get(), paf_score_data_.get(),
      paf_pair_index_data_.get(), paf_index_a_data_.get(),
      paf_index_b_data_.get(), pair_connections_count_,
      reinterpret_cast<float*>(peaks_data_.get()), max_peaks_,
      body_part_pairs_data_.get(), number_body_parts_);

  number_people_ = 0;
  remove_people_below_thresholds_and_fill_faces(
      valid_subset_indexes_data_.get(), number_people_,
      people_vector_body_data_.get(), people_vector_score_data_.get(),
      person_removed_data_.get(), people_vector_count_, max_peaks_,
      number_body_parts_, min_subset_cnt_, min_subset_score_,
      maximize_positives_);

  std::fill_n(pose_keypoints_ptr_,
              number_people_ * number_body_parts_ * peak_dim_, 0);
  std::fill_n(pose_scores_ptr_, number_people_, 0);
  people_vector_to_people_array(
      pose_keypoints_ptr_, pose_scores_ptr_, scale_factor_,
      people_vector_body_data_.get(), people_vector_score_data_.get(),
      valid_subset_indexes_data_.get(), number_people_,
      reinterpret_cast<float*>(peaks_data_.get()), number_body_parts_,
      number_body_part_pairs_);

  return number_people_;
}

}  // namespace openposert
