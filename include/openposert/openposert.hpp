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
 public:
  OpenPoseRT(const fs::path& engine_path, std::size_t input_width,
             std::size_t input_height, std::size_t input_channels,
             std::size_t net_input_width = 0, std::size_t net_input_height = 0,
             std::size_t max_joints = 25, std::size_t max_person = 128);

  Engine& engine() { return engine_; }

  auto get_input_data() { return input_data_.get(); }

  void forward();

  const auto& get_pose_keypoints() { return pose_keypoints_; }

 private:
  void malloc_memory();

  Engine engine_;

  std::size_t input_width_;
  std::size_t input_height_;
  std::size_t input_channels_;

  std::size_t net_input_width_;
  std::size_t net_input_height_;

  std::shared_ptr<void> input_data_;
  std::shared_ptr<void> normalized_data_;
  std::shared_ptr<void> net_input_data_;

  InputPreprocessing input_preprocessing_;

  std::shared_ptr<void> net_output_data_;

  float resize_factor_ = 4;
  std::shared_ptr<void> resize_net_output_data_;

  std::shared_ptr<void> kernel_data_;

  int max_joints_;
  int max_person_;
  int peak_dim_ = 3;

  std::array<int, 4> nms_source_size_;
  std::array<int, 4> nms_target_size_;
  Point<float> nms_offset_ = {0.5, 0.5};
  float nms_threshold_ = 0.05;

  std::shared_ptr<void> peaks_data_;

  float inter_min_above_threshold_ = 0.95;
  float inter_threshold_ = 0.05;
  int min_subset_cnt_ = 3;
  float min_subset_score_ = 0.4;

  PoseModel pose_model_ = PoseModel::BODY_25;

  std::vector<unsigned int> body_part_pair_;
  std::shared_ptr<void> body_part_pair_gpu_;

  std::vector<unsigned int> pose_map_idx_;
  std::shared_ptr<void> pose_map_idx_gpu_;

  Array<float> pair_scores_cpu_;
  std::shared_ptr<void> pair_scores_data_;

  Array<float> pose_keypoints_;
  Array<float> pose_scores_;
};

}  // namespace openposert
