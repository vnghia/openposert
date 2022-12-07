#include "minrt/utils.hpp"
#include "openposert/gpu/cuda.hpp"
#include "openposert/net/input_preprocessing.hpp"
#include "openposert/net/nms.hpp"
#include "openposert/net/output_postprocessing.hpp"
#include "openposert/net/resize_and_merge.hpp"
#include "openposert/openposert.hpp"
#include "openposert/pose/enum.hpp"
#include "openposert/pose/pose_parameters.hpp"
#include "openposert/utilities/fast_math.hpp"
#include "spdlog/fmt/bundled/format.h"
#include "spdlog/spdlog.h"
#include "thrust/execution_policy.h"
#include "thrust/fill.h"

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
      maximize_positive(maximize_positive),
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
              : get_pose_default_nms_threshold(pose_model, maximize_positive)),
      inter_min_above_threshold(
          (inter_min_above_threshold_in > 0)
              ? inter_min_above_threshold_in
              : get_pose_default_connect_inter_min_above_threshold(
                    maximize_positive)),
      inter_threshold((inter_threshold_in > 0)
                          ? inter_threshold_in
                          : get_pose_default_connect_inter_threshold(
                                pose_model, maximize_positive)),
      min_subset_cnt((min_subset_cnt_in > 0)
                         ? min_subset_cnt_in
                         : get_pose_default_min_subset_cnt(maximize_positive)),
      min_subset_score(
          (min_subset_score_in > 0)
              ? min_subset_score_in
              : get_pose_default_connect_min_subset_score(maximize_positive)),
      body_part_pair(get_pose_part_pairs(pose_model)),
      pose_map_idx(([&pose_model, maximize_positive]() {
        const auto number_body_part = get_pose_number_body_parts(pose_model);
        auto pose_map_idx = get_pose_map_index(pose_model);
        const auto offset = (add_bkg_channel(pose_model) ? 1 : 0);
        for (auto &i : pose_map_idx) i += (number_body_part + offset);
        return pose_map_idx;
      })()),
      number_body_parts(get_pose_number_body_parts(pose_model)),
      number_body_part_pairs(static_cast<int>(body_part_pair.size() / 2)) {
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

  auto body_part_pair_size = body_part_pair.size() * sizeof(unsigned int);
  body_part_pair_gpu_ = cuda_malloc_managed<unsigned int>(body_part_pair_size);
  cuda_upload(body_part_pair_gpu_.get(), body_part_pair.data(),
              body_part_pair_size);

  auto pose_map_idx_size = pose_map_idx.size() * sizeof(unsigned int);
  pose_map_idx_gpu_ = cuda_malloc_managed<unsigned int>(pose_map_idx_size);
  cuda_upload(pose_map_idx_gpu_.get(), pose_map_idx.data(), pose_map_idx_size);

  malloc_memory();
}

void OpenPoseRT::malloc_memory() {
  engine_.create_device_buffer(
      {{engine_.get_input_name(0), Engine::malloc_mode::managed},
       {engine_.get_output_name(0), Engine::malloc_mode::managed}});

  spdlog::info("malloc memory for OpenPoseRT");

  auto input_size =
      input_width * input_height * input_channels * sizeof(unsigned char);
  input_data_ = cuda_malloc_managed<unsigned char>(input_size);
  spdlog::info("allocated {} byte for input data with dim", input_size);

  net_input_data_ =
      std::static_pointer_cast<float>(engine_.get_input_device_owned_ptr(0));
  spdlog::info("use engine input buffer at {} for net input data",
               fmt::ptr(net_input_data_.get()));

  net_output_data_ =
      std::static_pointer_cast<float>(engine_.get_output_device_owned_ptr(0));
  spdlog::info("use engine output buffer at {} for net output data",
               fmt::ptr(net_output_data_.get()));

  input_preprocessing_ = InputPreprocessing(
      input_data_.get(), input_width, input_height, input_channels,
      net_input_data_.get(), net_input_width, net_input_height);

  auto pose_keypoints_size =
      max_person * number_body_parts * peak_dim * sizeof(float);
  pose_keypoints_data_ = cuda_malloc_managed<float>(pose_keypoints_size);
  spdlog::info("allocated {} byte for pose keypoints data",
               pose_keypoints_size);

  thrust::fill_n(thrust::host, static_cast<float *>(pose_keypoints_data_.get()),
                 max_person * number_body_parts * peak_dim, 0);

  auto pose_scores_size = max_person * sizeof(float);
  pose_scores_data_ = cuda_malloc_managed<float>(pose_scores_size);
  spdlog::info("allocated {} byte for pose scores data", pose_scores_size);

  thrust::fill_n(thrust::host, static_cast<float *>(pose_scores_data_.get()),
                 max_person, 0);

  output_postprocessing_ = OutputPostprocessing(
      pose_keypoints_data_.get(), pose_scores_data_.get(),
      net_output_data_.get(), static_cast<int>(net_output_dim.d[1]),
      net_output_width, net_output_height,
      fast_min(static_cast<float>(net_input_width) / net_output_width,
               static_cast<float>(net_input_height) / net_output_height),
      max_joints, peak_dim, nms_threshold, 1.f, number_body_parts,
      number_body_part_pairs, max_person, min_subset_cnt, min_subset_score,
      maximize_positive, inter_threshold, inter_min_above_threshold,
      nms_threshold, body_part_pair_gpu_.get(), pose_map_idx_gpu_.get());
}

void OpenPoseRT::forward() {
  input_preprocessing_.preprocessing_gpu();

  engine_.forward();
  cudaDeviceSynchronize();

  number_people_ = output_postprocessing_.postprocessing_gpu();
}

}  // namespace openposert
