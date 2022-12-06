#include "openposert/pose/pose_parameters.hpp"

#include <map>
#include <string>
#include <vector>

#include "openposert/pose/enum.hpp"

namespace openposert {

// body parts mapping
const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS{
    {0, "nose"},    {1, "neck"},       {2, "r_shoulder"},
    {3, "r_elbow"}, {4, "r_wrist"},    {5, "l_shoulder"},
    {6, "l_elbow"}, {7, "l_wrist"},    {8, "mid_hip"},
    {9, "r_hip"},   {10, "r_knee"},    {11, "r_ankle"},
    {12, "l_hip"},  {13, "l_knee"},    {14, "l_ankle"},
    {15, "r_eye"},  {16, "l_eye"},     {17, "r_ear"},
    {18, "l_ear"},  {19, "l_big_toe"}, {20, "l_small_toe"},
    {21, "l_heel"}, {22, "r_big_toe"}, {23, "r_small_toe"},
    {24, "r_heel"}, {25, "background"}};

const std::map<unsigned int, std::string> POSE_BODY_25B_BODY_PARTS{
    {0, "Nose"},       {1, "LEye"},       {2, "REye"},      {3, "LEar"},
    {4, "REar"},       {5, "LShoulder"},  {6, "RShoulder"}, {7, "LElbow"},
    {8, "RElbow"},     {9, "LWrist"},     {10, "RWrist"},   {11, "LHip"},
    {12, "RHip"},      {13, "LKnee"},     {14, "RKnee"},    {15, "LAnkle"},
    {16, "RAnkle"},    {17, "UpperNeck"}, {18, "HeadTop"},  {19, "LBigToe"},
    {20, "LSmallToe"}, {21, "LHeel"},     {22, "RBigToe"},  {23, "RSmallToe"},
    {24, "RHeel"},
};

const std::array<std::vector<unsigned int>, kNumModel> POSE_MAP_INDEX{
    // body_25
    std::vector<unsigned int>{
        0,  1,  14, 15, 22, 23, 16, 17, 18, 19, 24, 25, 26, 27, 6,  7,  2,  3,
        4,  5,  8,  9,  10, 11, 12, 13, 30, 31, 32, 33, 36, 37, 34, 35, 38, 39,
        20, 21, 28, 29, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51},
    // BODY_25B
    std::vector<unsigned int>{
        // Minimum spanning tree
        // |------------------------------------------- COCO Body
        // -------------------------------------------|
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        // Redundant ones
        // |------------------ Foot ------------------| |-- MPII --|
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        // Redundant ones
        // MPII redundant, ears, ears-shoulders, shoulders-wrists, wrists,
        // wrists-hips, hips, ankles)
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
        66, 67, 68, 69, 70, 71},
};

const std::array<std::map<unsigned int, std::string>, kNumModel>
    POSE_BODY_PART_MAPPING{POSE_BODY_25_BODY_PARTS, POSE_BODY_25B_BODY_PARTS};

const std::array<unsigned int, kNumModel> POSE_NUMBER_BODY_PARTS{25, 25};

const std::array<std::vector<unsigned int>, kNumModel> POSE_BODY_PART_PAIRS{
    // body_25
    std::vector<unsigned int>{
        1,  8,  1, 2,  1,  5,  2,  3,  3,  4,  5,  6,  6,  7,  8,  9,  9,  10,
        10, 11, 8, 12, 12, 13, 13, 14, 1,  0,  0,  15, 15, 17, 0,  16, 16, 18,
        2,  17, 5, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24},
    // BODY_25B
    std::vector<unsigned int>{
        // Minimum spanning tree
        // |------------------------------------------- COCO Body
        // -------------------------------------------|
        0, 1, 0, 2, 1, 3, 2, 4, 0, 5, 0, 6, 5, 7, 6, 8, 7, 9, 8, 10, 5, 11, 6,
        12, 11, 13, 12, 14, 13, 15, 14, 16,
        // |------------------ Foot ------------------| |-- MPII --|
        15, 19, 19, 20, 15, 21, 16, 22, 22, 23, 16, 24, 5, 17, 5, 18,
        // Redundant ones
        // MPII redundant, ears, ears-shoulders, shoulders-wrists, wrists,
        // wrists-hips, hips, ankles)
        6, 17, 6, 18, 3, 4, 3, 5, 4, 6, 5, 9, 6, 10, 9, 10, 9, 11, 10, 12, 11,
        12, 15, 16},
};

const std::map<unsigned int, std::string>& get_pose_body_part_mapping(
    const PoseModel pose_model) {
  return POSE_BODY_PART_MAPPING.at((int)pose_model);
}

unsigned int get_pose_number_body_parts(const PoseModel pose_model) {
  return POSE_NUMBER_BODY_PARTS.at((int)pose_model);
}

const std::vector<unsigned int>& get_pose_part_pairs(
    const PoseModel pose_model) {
  return POSE_BODY_PART_PAIRS.at((int)pose_model);
}

const std::vector<unsigned int>& get_pose_map_index(
    const PoseModel pose_model) {
  return POSE_MAP_INDEX.at((int)pose_model);
}

unsigned int get_pose_max_peaks() { return POSE_MAX_PEOPLE; }

float get_pose_net_decrease_factor(const PoseModel pose_model) { return 8.f; }

unsigned int pose_body_part_map_string_to_key(
    const PoseModel pose_model, const std::vector<std::string>& strings) {
  const auto& pose_body_part_mapping = POSE_BODY_PART_MAPPING[(int)pose_model];
  for (const auto& string : strings)
    for (const auto& pair : pose_body_part_mapping)
      if (pair.second == string) return pair.first;
  return 0;
}

unsigned int pose_body_part_map_string_to_key(const PoseModel pose_model,
                                              const std::string& string) {
  return pose_body_part_map_string_to_key(pose_model,
                                          std::vector<std::string>{string});
}

// default model parameters
// they might be modified on running time
float get_pose_default_nms_threshold(const PoseModel pose_model,
                                     const bool maximize_positives) {
  return (maximize_positives ? 0.02f : 0.05f);
}

float get_pose_default_connect_inter_min_above_threshold(
    const bool maximize_positives) {
  return (maximize_positives ? 0.75f : 0.95f);
}

float get_pose_default_connect_inter_threshold(const PoseModel pose_model,
                                               const bool maximize_positives) {
  // return (maximize_positives ? 0.01f : 0.5f); // 0.485 but much less
  // false positive connections return (maximize_positives ? 0.01f : 0.1f);
  // // 0.518 return (maximize_positives ? 0.01f : 0.075f); // 0.521
  return (maximize_positives ? 0.01f : 0.05f);  // 0.523
  // return (maximize_positives ? 0.01f : 0.01f); // 0.527 but huge amount of
  // false positives joints
}

unsigned int get_pose_default_min_subset_cnt(const bool maximize_positives) {
  return (maximize_positives ? 2u : 3u);
}

float get_pose_default_connect_min_subset_score(const bool maximize_positives) {
  return (maximize_positives ? 0.05f : 0.4f);
}

bool add_bkg_channel(const PoseModel pose_model) {
  return (POSE_BODY_PART_MAPPING[(int)pose_model].size() !=
          POSE_NUMBER_BODY_PARTS[(int)pose_model]);
}

}  // namespace openposert
