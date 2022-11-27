#include "openposert/pose/pose_parameters.hpp"

#include <map>
#include <string>
#include <vector>

#include "openposert/core/common.hpp"
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

const std::array<std::vector<unsigned int>, kNumModel> POSE_MAP_INDEX{
    // body_25
    std::vector<unsigned int>{
        0,  1,  14, 15, 22, 23, 16, 17, 18, 19, 24, 25, 26, 27, 6,  7,  2,  3,
        4,  5,  8,  9,  10, 11, 12, 13, 30, 31, 32, 33, 36, 37, 34, 35, 38, 39,
        20, 21, 28, 29, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51},
};

const std::array<std::map<unsigned int, std::string>, kNumModel>
    POSE_BODY_PART_MAPPING{POSE_BODY_25_BODY_PARTS};

const std::array<unsigned int, kNumModel> POSE_NUMBER_BODY_PARTS{25};

const std::array<std::vector<unsigned int>, kNumModel> POSE_BODY_PART_PAIRS{
    // body_25
    std::vector<unsigned int>{
        1,  8,  1, 2,  1,  5,  2,  3,  3,  4,  5,  6,  6,  7,  8,  9,  9,  10,
        10, 11, 8, 12, 12, 13, 13, 14, 1,  0,  0,  15, 15, 17, 0,  16, 16, 18,
        2,  17, 5, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24},
};

const std::map<unsigned int, std::string>& get_pose_body_part_mapping(
    const PoseModel pose_model) {
  try {
    return POSE_BODY_PART_MAPPING.at((int)pose_model);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return POSE_BODY_PART_MAPPING[(int)pose_model];
  }
}

unsigned int get_pose_number_body_parts(const PoseModel pose_model) {
  try {
    return POSE_NUMBER_BODY_PARTS.at((int)pose_model);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0u;
  }
}

const std::vector<unsigned int>& get_pose_part_pairs(
    const PoseModel pose_model) {
  try {
    return POSE_BODY_PART_PAIRS.at((int)pose_model);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return POSE_BODY_PART_PAIRS[(int)pose_model];
  }
}

const std::vector<unsigned int>& get_pose_map_index(
    const PoseModel pose_model) {
  try {
    return POSE_MAP_INDEX.at((int)pose_model);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return POSE_MAP_INDEX[(int)pose_model];
  }
}

unsigned int get_pose_max_peaks() {
  try {
    return POSE_MAX_PEOPLE;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0u;
  }
}

float get_pose_net_decrease_factor(const PoseModel pose_model) {
  try {
    return 8.f;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0.f;
  }
}

unsigned int pose_body_part_map_string_to_key(
    const PoseModel pose_model, const std::vector<std::string>& strings) {
  try {
    const auto& pose_body_part_mapping =
        POSE_BODY_PART_MAPPING[(int)pose_model];
    for (const auto& string : strings)
      for (const auto& pair : pose_body_part_mapping)
        if (pair.second == string) return pair.first;
    error("string(s) could not be found.", __LINE__, __FUNCTION__, __FILE__);
    return 0;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0;
  }
}

unsigned int pose_body_part_map_string_to_key(const PoseModel pose_model,
                                              const std::string& string) {
  try {
    return pose_body_part_map_string_to_key(pose_model,
                                            std::vector<std::string>{string});
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0;
  }
}

// default model parameters
// they might be modified on running time
float get_pose_default_nms_threshold(const PoseModel pose_model,
                                     const bool maximize_positives) {
  try {
    return (maximize_positives ? 0.02f : 0.05f);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0.f;
  }
}

float get_pose_default_connect_inter_min_above_threshold(
    const bool maximize_positives) {
  try {
    return (maximize_positives ? 0.75f : 0.95f);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0.f;
  }
}

float get_pose_default_connect_inter_threshold(const PoseModel pose_model,
                                               const bool maximize_positives) {
  try {
    // return (maximize_positives ? 0.01f : 0.5f); // 0.485 but much less
    // false positive connections return (maximize_positives ? 0.01f : 0.1f);
    // // 0.518 return (maximize_positives ? 0.01f : 0.075f); // 0.521
    return (maximize_positives ? 0.01f : 0.05f);  // 0.523
    // return (maximize_positives ? 0.01f : 0.01f); // 0.527 but huge amount of
    // false positives joints
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0.f;
  }
}

unsigned int get_pose_default_min_subset_cnt(const bool maximize_positives) {
  try {
    return (maximize_positives ? 2u : 3u);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0u;
  }
}

float get_pose_default_connect_min_subset_score(const bool maximize_positives) {
  try {
    return (maximize_positives ? 0.05f : 0.4f);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0.f;
  }
}

bool add_bkg_channel(const PoseModel pose_model) {
  try {
    return (POSE_BODY_PART_MAPPING[(int)pose_model].size() !=
            POSE_NUMBER_BODY_PARTS[(int)pose_model]);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return false;
  }
}

}  // namespace openposert
