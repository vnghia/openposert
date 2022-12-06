#pragma once

#include <filesystem>
#include <memory>

#include "minrt/minrt.hpp"
#include "openposert/net/input_preprocessing.hpp"
#include "openposert/net/output_postprocessing.hpp"
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
    return number_people_ * number_body_parts * peak_dim;
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
  const int net_output_width;
  const int net_output_height;

  const float resize_factor;

  const int max_joints;
  const int max_person;

  // x y score
  const int peak_dim = 3;

  const std::array<int, 4> nms_source_size;
  const std::array<int, 4> nms_target_size;
  const float nms_offset_x = 0.5;
  const float nms_offset_y = 0.5;

  const float nms_threshold;

  const float inter_min_above_threshold;
  const float inter_threshold;
  const unsigned int min_subset_cnt;
  const float min_subset_score;

  const std::vector<unsigned int> body_part_pair;
  const std::vector<unsigned int> pose_map_idx;

  int number_body_parts;
  int number_body_part_pairs;

 private:
  void malloc_memory();

  std::shared_ptr<unsigned int> body_part_pair_gpu_;
  std::shared_ptr<unsigned int> pose_map_idx_gpu_;

  std::shared_ptr<unsigned char> input_data_;
  std::shared_ptr<float> net_input_data_;

  InputPreprocessing input_preprocessing_;

  std::shared_ptr<float> net_output_data_;

  std::shared_ptr<int> kernel_data_;

  std::shared_ptr<float> peaks_data_;

  std::shared_ptr<float> pair_scores_data_;

  int number_people_;

  std::shared_ptr<float> pose_keypoints_data_;
  std::shared_ptr<float> pose_scores_data_;

  OutputPostprocessing output_postprocessing_;
};

}  // namespace openposert
