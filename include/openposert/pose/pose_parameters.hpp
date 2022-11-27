#pragma once

#include <map>
#include <string>

#include "openposert/core/common.hpp"
#include "openposert/pose/enum.hpp"

namespace openposert {
const auto POSE_MAX_PEOPLE = 127u;

const std::map<unsigned int, std::string>& get_pose_body_part_mapping(
    const PoseModel pose_model);
unsigned int get_pose_number_body_parts(const PoseModel pose_model);

const std::vector<unsigned int>& get_pose_part_pairs(
    const PoseModel pose_model);
const std::vector<unsigned int>& get_pose_map_index(const PoseModel pose_model);
unsigned int get_pose_max_peaks();

float get_pose_net_decrease_factor(const PoseModel pose_model);

unsigned int pose_body_part_map_string_to_key(const PoseModel pose_model,
                                              const std::string& string);
unsigned int pose_body_part_map_string_to_key(
    const PoseModel pose_model, const std::vector<std::string>& strings);

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
