#pragma once

#include <utility>
#include <vector>

#include "openposert/core/array.hpp"
#include "openposert/core/point.hpp"
#include "openposert/pose/enum.hpp"

namespace openposert {

template <typename T>
void connect_body_parts_cpu(
    Array<T>& pose_keypoints, Array<T>& pose_scores,
    const T* const heat_map_ptr, const T* const peaks_ptr,
    const PoseModel pose_model, const Point<int>& heat_map_size,
    const int max_peaks, const T inter_min_above_threshold,
    const T inter_threshold, const int min_subset_cnt, const T min_subset_score,
    const T default_nms_threshold, const T scale_factor = 1.f,
    const bool maximize_positives = false);

template <typename T>
void connect_body_parts_gpu(
    Array<T>& pose_keypoints, Array<T>& pose_scores,
    const T* const heat_map_gpu_ptr, const T* const peaks_ptr,
    const PoseModel pose_model, const Point<int>& heat_map_size,
    const int max_peaks, const T inter_min_above_threshold,
    const T inter_threshold, const int min_subset_cnt, const T min_subset_score,
    const T default_nms_threshold, const T scale_factor,
    const bool maximize_positives, Array<T> pair_scores_cpu,
    T* pair_scores_gpu_ptr, const unsigned int* const body_part_pairs_gpu_ptr,
    const unsigned int* const map_idx_gpu_ptr, const T* const peaks_gpu_ptr);

// private functions used by the 2 above functions
template <typename T>
std::vector<std::pair<std::vector<int>, T>> create_people_vector(
    const T* const heat_map_ptr, const T* const peaks_ptr,
    const PoseModel pose_model, const Point<int>& heat_map_size,
    const int max_peaks, const T inter_threshold,
    const T inter_min_above_threshold,
    const std::vector<unsigned int>& body_part_pairs,
    const unsigned int number_body_parts,
    const unsigned int number_body_part_pairs, const T default_nms_threshold,
    const Array<T>& precomputed_pa_fs = Array<T>());

template <typename T>
void remove_people_below_thresholds_and_fill_faces(
    std::vector<int>& valid_subset_indexes, int& number_people,
    std::vector<std::pair<std::vector<int>, T>>& subsets,
    const unsigned int number_body_parts, const int min_subset_cnt,
    const T min_subset_score, const bool maximize_positives,
    const T* const peaks_ptr);

template <typename T>
void people_vector_to_people_array(
    Array<T>& pose_keypoints, Array<T>& pose_scores, const T scale_factor,
    const std::vector<std::pair<std::vector<int>, T>>& subsets,
    const std::vector<int>& valid_subset_indexes, const T* const peaks_ptr,
    const int number_people, const unsigned int number_body_parts,
    const unsigned int number_body_part_pairs);

template <typename T>
std::vector<std::tuple<T, T, int, int, int>> paf_ptr_into_vector(
    const Array<T>& pair_scores, const T* const peaks_ptr, const int max_peaks,
    const std::vector<unsigned int>& body_part_pairs,
    const unsigned int number_body_part_pairs);

template <typename T>
std::vector<std::pair<std::vector<int>, T>> paf_vector_into_people_vector(
    const std::vector<std::tuple<T, T, int, int, int>>& pair_scores,
    const T* const peaks_ptr, const int max_peaks,
    const std::vector<unsigned int>& body_part_pairs,
    const unsigned int number_body_parts);

}  // namespace openposert
