#include "openposert/utilities/pose_model.hpp"

#include <array>
#include <vector>

namespace openposert {

static const std::array<std::vector<unsigned int>, kNumPoseModel>
    POSE_MAP_INDEX{
        // BODY_25
        std::vector<unsigned int>{0,  1,  14, 15, 22, 23, 16, 17, 18, 19, 24,
                                  25, 26, 27, 6,  7,  2,  3,  4,  5,  8,  9,
                                  10, 11, 12, 13, 30, 31, 32, 33, 36, 37, 34,
                                  35, 38, 39, 20, 21, 28, 29, 40, 41, 42, 43,
                                  44, 45, 46, 47, 48, 49, 50, 51},
        // BODY_25B
        std::vector<unsigned int>{
            0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
            45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71},
    };

static const std::array<unsigned int, kNumPoseModel> POSE_NUMBER_BODY_PARTS{25,
                                                                            25};

const std::array<std::vector<unsigned int>, kNumPoseModel> POSE_BODY_PART_PAIRS{
    // BODY_25
    std::vector<unsigned int>{
        1,  8,  1, 2,  1,  5,  2,  3,  3,  4,  5,  6,  6,  7,  8,  9,  9,  10,
        10, 11, 8, 12, 12, 13, 13, 14, 1,  0,  0,  15, 15, 17, 0,  16, 16, 18,
        2,  17, 5, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24},
    // BODY_25B
    std::vector<unsigned int>{
        0,  1,  0,  2,  1,  3,  2,  4,  0,  5,  0,  6,  5,  7,  6,  8,  7,  9,
        8,  10, 5,  11, 6,  12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 19, 19, 20,
        15, 21, 16, 22, 22, 23, 16, 24, 5,  17, 5,  18, 6,  17, 6,  18, 3,  4,
        3,  5,  4,  6,  5,  9,  6,  10, 9,  10, 9,  11, 10, 12, 11, 12, 15, 16},
};

unsigned int get_pose_number_body_parts(const PoseModel pose_model) {
  return POSE_NUMBER_BODY_PARTS[static_cast<uint8_t>(pose_model)];
}

const std::vector<unsigned int>& get_pose_part_pairs(
    const PoseModel pose_model) {
  return POSE_BODY_PART_PAIRS[static_cast<uint8_t>(pose_model)];
}

const std::vector<unsigned int>& get_pose_map_index(
    const PoseModel pose_model) {
  return POSE_MAP_INDEX[static_cast<uint8_t>(pose_model)];
}

unsigned int get_pose_max_peaks() { return kMaxPeople; }

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
  return (maximize_positives ? 0.01f : 0.05f);
}

unsigned int get_pose_default_min_subset_cnt(const bool maximize_positives) {
  return (maximize_positives ? 2u : 3u);
}

float get_pose_default_connect_min_subset_score(const bool maximize_positives) {
  return (maximize_positives ? 0.05f : 0.4f);
}

bool add_bkg_channel(const PoseModel pose_model) {
  return pose_model == PoseModel::BODY_25;
}

}  // namespace openposert
