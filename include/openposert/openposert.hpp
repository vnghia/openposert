#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>

#include "half.hpp"
#include "minrt/minrt.hpp"
#include "openposert/input/input.hpp"
#include "openposert/output/output.hpp"
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
             bool maximize_positives = false, float resize_factor_in = 0,
             int max_joints_in = 0, int max_person_in = 0,
             float nms_threshold_in = 0, float inter_min_above_threshold_in = 0,
             float inter_threshold_in = 0, unsigned int min_subset_cnt_in = 0,
             float min_subset_score_in = 0);

  Engine& engine() { return engine_; }

  auto get_input_data() { return input_data_.get(); }

  void forward();

  const auto get_pose_keypoints() { return pose_keypoints_data_.get(); }

  const auto get_pose_keypoints_size() {
    return number_people_ * number_body_parts * peak_dim;
  }

  const int input_width;
  const int input_height;
  const int input_channels;

  const PoseModel pose_model;

  bool maximize_positives;

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

  const float nms_threshold;

  const float inter_min_above_threshold;
  const float inter_threshold;
  const unsigned int min_subset_cnt;
  const float min_subset_score;

  int number_body_parts;

 private:
  void malloc_memory();

  std::shared_ptr<uint8_t> input_data_;
  std::shared_ptr<__half> net_input_data_;

  Input input_;

  std::shared_ptr<__half> net_output_data_;

  int number_people_;

  std::shared_ptr<half_float::half[]> pose_keypoints_data_;
  std::shared_ptr<half_float::half[]> pose_scores_data_;

  Output output_;
};

}  // namespace openposert
