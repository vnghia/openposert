#pragma once

#include <filesystem>
#include <memory>

#include "minrt/minrt.hpp"
#include "openposert/core/array.hpp"
#include "openposert/core/point.hpp"
#include "openposert/net/input_preprocessing.hpp"
#include "openposert/pose/enum.hpp"
#include "openposert/pose/pose_parameters.hpp"

namespace openposert {

using namespace minrt;

class OpenPoseRT {
 private:
  Engine engine_;

 public:
  OpenPoseRT(const fs::path& engine_path, int input_width, int input_height,
             int input_channels, PoseModel pose_model = PoseModel::BODY_25,
             bool maximize_positive = false, float resize_factor_in = 0,
             int max_joints_in = 0, int max_person_in = 0,
             float nms_threshold_in = 0, float inter_min_above_threshold_in = 0,
             float inter_threshold_in = 0, unsigned int min_subset_cnt_in = 0,
             float min_subset_score_in = 0);

  Engine& engine() { return engine_; }

  auto get_input_data() { return input_data_.get(); }

  void forward();

  const auto get_pose_keypoints() {
    return static_cast<float*>(pose_keypoints_data_.get());
  }

  const auto get_pose_keypoints_size() {
    return number_people_ * get_pose_number_body_parts(pose_model) * peak_dim;
  }

  const int input_width;
  const int input_height;
  const int input_channels;

  const PoseModel pose_model;

  bool maximize_positive;

  const nvinfer1::Dims net_input_dim;

  const int net_input_width;
  const int net_input_height;

  const nvinfer1::Dims net_output_dim;

  const float resize_factor;

  const int max_joints;
  const int max_person;

  // x y score
  const int peak_dim = 3;

  const std::array<int, 4> nms_source_size;
  const std::array<int, 4> nms_target_size;
  const Point<float> nms_offset = {0.5, 0.5};

  const float nms_threshold;

  const Point<int> body_part_net_output_size;

  const float inter_min_above_threshold;
  const float inter_threshold;
  const unsigned int min_subset_cnt;
  const float min_subset_score;

  const std::vector<unsigned int> body_part_pair;
  const std::vector<unsigned int> pose_map_idx;

 private:
  void malloc_memory();

  std::shared_ptr<void> body_part_pair_gpu_;
  std::shared_ptr<void> pose_map_idx_gpu_;

  std::shared_ptr<void> input_data_;
  std::shared_ptr<void> normalized_data_;
  std::shared_ptr<void> net_input_data_;

  InputPreprocessing input_preprocessing_;

  std::shared_ptr<void> net_output_data_;

  std::shared_ptr<void> resize_net_output_data_;

  std::shared_ptr<void> kernel_data_;

  std::shared_ptr<void> peaks_data_;

  std::shared_ptr<void> pair_scores_data_;

  int number_people_;

  std::shared_ptr<void> pose_keypoints_data_;
  std::shared_ptr<void> pose_scores_data_;
};

}  // namespace openposert
