#include <algorithm>
#include <iostream>

#include "cuda_fp16.h"
#include "minrt/utils.hpp"
#include "openposert/input/input.hpp"
#include "openposert/openposert.hpp"
#include "openposert/output/output.hpp"
#include "openposert/utilities/cuda.hpp"
#include "openposert/utilities/pose_model.hpp"
#include "spdlog/fmt/bundled/ranges.h"
#include "spdlog/spdlog.h"

namespace openposert {

OpenPoseRT::OpenPoseRT(const fs::path &engine_path, int input_width,
                       int input_height, int input_channels,
                       PoseModel pose_model, bool maximize_positive,
                       float resize_factor_in, int max_joints_in,
                       int max_person_in, float nms_threshold_in,
                       float inter_min_above_threshold_in,
                       float inter_threshold_in, unsigned int min_subset_cnt_in,
                       float min_subset_score_in)
    : engine_(Engine::engine_from_path(engine_path)),
      input_width(input_width),
      input_height(input_height),
      input_channels(input_channels),
      pose_model(pose_model),
      maximize_positives(maximize_positives),
      net_input_dim(engine_.get_input_dim(0)),
      net_input_width(net_input_dim.d[3]),
      net_input_height(net_input_dim.d[2]),
      net_output_dim(engine_.get_output_dim(0)),
      net_output_width(net_output_dim.d[3]),
      net_output_height(net_output_dim.d[2]),
      resize_factor((resize_factor_in > 0) ? resize_factor_in : 1),
      max_joints((max_joints_in > 0) ? max_joints_in
                                     : get_pose_number_body_parts(pose_model)),
      max_person((max_person_in > 0) ? max_person_in : 127),
      nms_threshold(
          (nms_threshold_in > 0)
              ? nms_threshold_in
              : get_pose_default_nms_threshold(pose_model, maximize_positives)),
      inter_min_above_threshold(
          (inter_min_above_threshold_in > 0)
              ? inter_min_above_threshold_in
              : get_pose_default_connect_inter_min_above_threshold(
                    maximize_positives)),
      inter_threshold((inter_threshold_in > 0)
                          ? inter_threshold_in
                          : get_pose_default_connect_inter_threshold(
                                pose_model, maximize_positives)),
      min_subset_cnt((min_subset_cnt_in > 0)
                         ? min_subset_cnt_in
                         : get_pose_default_min_subset_cnt(maximize_positives)),
      min_subset_score(
          (min_subset_score_in > 0)
              ? min_subset_score_in
              : get_pose_default_connect_min_subset_score(maximize_positives)),
      number_body_parts(get_pose_number_body_parts(pose_model)) {
  spdlog::info("OpenPoseRT image input width = {}, height = {}, channel = {}",
               input_width, input_height, input_channels);
  spdlog::info("OpenPoseRT net input width = {}, height = {}", net_input_width,
               net_input_height);
  spdlog::info(
      "OpenposeRT parameters maximize_positive={}, resize_factor={}, "
      "max_joints={}, max_person={}, "
      "nms_threshold={}, inter_min_above_threshold={}, inter_threshold={}, "
      "min_subset_cnt={}, min_subset_score={}",
      maximize_positive, resize_factor, max_joints, max_person, nms_threshold,
      inter_min_above_threshold, inter_threshold, min_subset_cnt,
      min_subset_score);

  malloc_memory();
}

void OpenPoseRT::malloc_memory() {
  engine_.create_device_buffer(
      {{engine_.get_input_name(0), Engine::malloc_mode::managed},
       {engine_.get_output_name(0), Engine::malloc_mode::managed}});

  spdlog::info("malloc memory for OpenPoseRT");

  auto input_size =
      input_width * input_height * input_channels * sizeof(uint8_t);
  input_data_ = cuda_malloc_managed<uint8_t>(input_size);
  spdlog::info("allocated {} byte for input data with dim", input_size);

  net_input_data_ =
      std::static_pointer_cast<__half>(engine_.get_input_device_owned_ptr(0));
  spdlog::info("use engine input buffer at {} for net input data",
               fmt::ptr(net_input_data_.get()));

  net_output_data_ =
      std::static_pointer_cast<float>(engine_.get_output_device_owned_ptr(0));
  spdlog::info("use engine output buffer at {} for net output data",
               fmt::ptr(net_output_data_.get()));

  input_ = Input(input_data_.get(), input_width, input_height, input_channels,
                 net_input_data_.get(), net_input_width, net_input_height);

  auto pose_keypoints_size = max_person * number_body_parts * peak_dim;
  pose_keypoints_data_.reset(new float[pose_keypoints_size]);
  spdlog::info("allocated {} byte for pose keypoints data",
               pose_keypoints_size);
  std::fill_n(pose_keypoints_data_.get(), pose_keypoints_size, 0);

  auto pose_scores_size = max_person;
  pose_scores_data_.reset(new float[pose_scores_size]);
  spdlog::info("allocated {} byte for pose scores data", pose_scores_size);
  std::fill_n(pose_scores_data_.get(), pose_scores_size, 0);

  output_ =
      Output(pose_keypoints_data_.get(), pose_scores_data_.get(),
             static_cast<float>(1), peak_dim, net_output_data_.get(),
             net_output_width, net_output_height,
             static_cast<int>(net_output_dim.d[1]), max_joints, max_person,
             pose_model, maximize_positives, static_cast<__half>(nms_threshold),
             static_cast<__half>(inter_min_above_threshold),
             static_cast<__half>(inter_threshold), min_subset_cnt,
             static_cast<float>(min_subset_score));
}

void OpenPoseRT::forward() {
  input_.process();

  engine_.forward();
  cudaDeviceSynchronize();

  number_people_ = output_.process();
}

}  // namespace openposert
