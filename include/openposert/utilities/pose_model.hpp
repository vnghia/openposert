#pragma once

#include <cstdint>
#include <vector>

namespace openposert {

constexpr unsigned int kNumPoseModel = 2;
enum class PoseModel : uint8_t { BODY_25 = 0, BODY_25B };

constexpr unsigned int kMaxPeople = 127;

unsigned int get_pose_number_body_parts(const PoseModel pose_model);

const std::vector<unsigned int>& get_pose_part_pairs(
    const PoseModel pose_model);

const std::vector<unsigned int>& get_pose_map_index(const PoseModel pose_model);

unsigned int get_pose_max_peaks();

float get_pose_default_nms_threshold(const PoseModel pose_model,
                                     const bool maximize_positives);

float get_pose_default_connect_inter_min_above_threshold(
    const bool maximize_positives);

float get_pose_default_connect_inter_threshold(const PoseModel pose_model,
                                               const bool maximize_positives);

unsigned int get_pose_default_min_subset_cnt(const bool maximize_positives);

float get_pose_default_connect_min_subset_score(const bool maximize_positives);

bool add_bkg_channel(const PoseModel pose_model);

}  // namespace openposert
