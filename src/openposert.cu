#include "openposert/core/array.hpp"
#include "openposert/core/point.hpp"
#include "openposert/gpu/cuda.hpp"
#include "openposert/net/body_part_connector.hpp"
#include "openposert/net/input_preprocessing.hpp"
#include "openposert/net/nms.hpp"
#include "openposert/net/resize_and_merge.hpp"
#include "openposert/openposert.hpp"
#include "openposert/pose/enum.hpp"
#include "openposert/pose/pose_parameters.hpp"
#include "spdlog/fmt/bundled/format.h"
#include "spdlog/spdlog.h"

namespace openposert {

OpenPoseRT::OpenPoseRT(const fs::path &engine_path, std::size_t input_width,
                       std::size_t input_height, std::size_t input_channels,
                       std::size_t net_input_width,
                       std::size_t net_input_height, std::size_t max_joints,
                       std::size_t max_person)
    : engine_(Engine::engine_from_path(engine_path)),
      input_width_(input_width),
      input_height_(input_height),
      input_channels_(input_channels),
      max_joints_(max_joints),
      max_person_(max_person) {
  net_input_height_ = (net_input_width != 0) ? net_input_height_
                                             : engine_.get_input_dim(0).d[2];
  net_input_width_ = (net_input_height != 0) ? net_input_width_
                                             : engine_.get_input_dim(0).d[3];

  spdlog::info("OpenPoseRT input_dim = ({}, {}, {}) net_input_dim = ({}, {})",
               input_width_, input_height_, input_channels_, net_input_width_,
               net_input_height_);

  body_part_pair_ = get_pose_part_pairs(pose_model_);
  auto body_part_pair_size =
      body_part_pair_.size() * sizeof(decltype(body_part_pair_)::value_type);
  body_part_pair_gpu_ = cuda_malloc(body_part_pair_size);
  cuda_upload(body_part_pair_gpu_.get(), body_part_pair_.data(),
              body_part_pair_size);

  const auto number_body_part = get_pose_number_body_parts(pose_model_);
  pose_map_idx_ = get_pose_map_index(pose_model_);
  const auto offset = (add_bkg_channel(pose_model_) ? 1 : 0);
  for (auto &i : pose_map_idx_) i += (number_body_part + offset);

  auto pose_map_idx_size =
      pose_map_idx_.size() * sizeof(decltype(pose_map_idx_)::value_type);
  pose_map_idx_gpu_ = cuda_malloc(pose_map_idx_size);
  cuda_upload(pose_map_idx_gpu_.get(), pose_map_idx_.data(), pose_map_idx_size);

  malloc_memory();
}

void OpenPoseRT::malloc_memory() {
  engine_.create_device_buffer(
      {{engine_.get_input_name(0), Engine::malloc_mode::managed},
       {engine_.get_output_name(0), Engine::malloc_mode::managed}});

  spdlog::info("malloc memory for OpenPoseRT");

  auto input_size =
      input_width_ * input_height_ * input_channels_ * sizeof(unsigned char);
  input_data_ = cuda_malloc_managed(input_size);
  spdlog::info("allocated {} byte for input data with dim", input_size);

  auto normalized_input_size =
      input_width_ * input_height_ * input_channels_ * sizeof(float);
  normalized_data_ = cuda_malloc_managed(normalized_input_size);
  spdlog::info("allocated {} byte for normalized input data",
               normalized_input_size);

  net_input_data_ = engine_.get_input_device_owned_ptr(0);
  spdlog::info("use engine input buffer at {} for net input data",
               net_input_data_.get());

  net_output_data_ = engine_.get_output_device_owned_ptr(0);
  spdlog::info("use engine output buffer at {} for net output data",
               net_output_data_.get());

  input_preprocessing_ = InputPreprocessing(
      static_cast<unsigned char *>(input_data_.get()),
      static_cast<float *>(normalized_data_.get()), input_width_, input_height_,
      input_channels_, static_cast<float *>(net_input_data_.get()),
      net_input_width_, net_input_height_);

  auto resize_net_output_size =
      engine_.get_output_size(0) * resize_factor_ * resize_factor_;
  resize_net_output_data_ = cuda_malloc_managed(resize_net_output_size);
  spdlog::info("allocated {} byte for resized net output data",
               resize_net_output_size);

  auto output_dims = engine_.get_output_dim(0);
  auto kernel_size = get_total_size(output_dims) * sizeof(int);
  kernel_data_ = cuda_malloc_managed(kernel_size);
  spdlog::info("allocated {} byte for kernel data", kernel_size);

  nms_source_size_ = {
      static_cast<int>(output_dims.d[0]), static_cast<int>(output_dims.d[1]),
      static_cast<int>(output_dims.d[2]), static_cast<int>(output_dims.d[3])};
  nms_target_size_ = {1, max_joints_, max_person_ + 1, peak_dim_};

  auto peaks_size = max_joints_ * (max_person_ + 1) * peak_dim_ * sizeof(float);
  peaks_data_ = cuda_malloc_managed(peaks_size);
  spdlog::info("allocated {} byte for peaks data", peaks_size);

  auto number_body_part_pairs = static_cast<int>(body_part_pair_.size() / 2);
  auto peaks_score_size =
      number_body_part_pairs * max_person_ * max_person_ * sizeof(float);
  pair_scores_data_ = cuda_malloc_managed(peaks_score_size);
  spdlog::info("allocated {} byte for peaks score data", peaks_score_size);

  auto pose_keypoints_size = max_person_ *
                             get_pose_number_body_parts(pose_model_) *
                             peak_dim_ * sizeof(float);
  pose_keypoints_data_ = cuda_malloc_managed(pose_keypoints_size);
  spdlog::info("allocated {} byte for pose keypoints data",
               pose_keypoints_size);

  auto pose_scores_size = max_person_ * sizeof(float);
  pose_scores_data_ = cuda_malloc_managed(pose_scores_size);
  spdlog::info("allocated {} byte for pose scores data", pose_scores_size);
}

void OpenPoseRT::forward() {
  input_preprocessing_.preprocessing_gpu();

  engine_.forward();
  cudaDeviceSynchronize();

  nms_gpu(static_cast<float *>(peaks_data_.get()),
          static_cast<int *>(kernel_data_.get()),
          static_cast<float *>(net_output_data_.get()), nms_threshold_,
          nms_target_size_, nms_source_size_, nms_offset_);

  Point<int> net_output_size = {engine_.get_output_dim(0).d[3],
                                engine_.get_output_dim(0).d[2]};

  connect_body_parts_gpu(
      static_cast<float *>(pose_keypoints_data_.get()),
      static_cast<float *>(pose_scores_data_.get()), number_people_,
      static_cast<float *>(net_output_data_.get()),
      static_cast<float *>(peaks_data_.get()), pose_model_, net_output_size,
      max_person_, inter_min_above_threshold_, inter_threshold_,
      min_subset_cnt_, min_subset_score_, nms_threshold_, 1.f, false,
      static_cast<float *>(pair_scores_data_.get()),
      static_cast<unsigned int *>(body_part_pair_gpu_.get()),
      static_cast<unsigned int *>(pose_map_idx_gpu_.get()),
      static_cast<float *>(peaks_data_.get()));
}

}  // namespace openposert
