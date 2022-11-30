#include "openposert/core/common.hpp"
#include "openposert/gpu/cuda.hpp"
#include "openposert/gpu/cuda_fast_math.hpp"
#include "openposert/net/body_part_connector.hpp"
#include "openposert/pose/pose_parameters.hpp"
#include "openposert/utilities/fast_math.hpp"

namespace openposert {

template <typename T>
inline __device__ T process(const T* body_part_a, const T* body_part_b,
                            const T* map_x, const T* map_y,
                            const int heatmap_width, const int heatmap_height,
                            const T inter_threshold,
                            const T inter_min_above_threshold,
                            const T default_nms_threshold) {
  const auto vector_a_to_bx = body_part_b[0] - body_part_a[0];
  const auto vector_a_to_by = body_part_b[1] - body_part_a[1];
  const auto vector_a_to_b_max = max(abs(vector_a_to_bx), abs(vector_a_to_by));
  const auto number_points_in_line =
      max(5, min(25, positive_int_round_cuda(sqrt(5 * vector_a_to_b_max))));
  const auto vector_norm = T(
      sqrt(vector_a_to_bx * vector_a_to_bx + vector_a_to_by * vector_a_to_by));

  // if the peaks_ptr are coincident. don'T connect them.
  if (vector_norm > 1e-6) {
    const auto s_x = body_part_a[0];
    const auto s_y = body_part_a[1];
    const auto vector_a_to_b_norm_x = vector_a_to_bx / vector_norm;
    const auto vector_a_to_b_norm_y = vector_a_to_by / vector_norm;

    auto sum = T(0.);
    auto count = 0;
    const auto vector_a_to_bx_in_line = vector_a_to_bx / number_points_in_line;
    const auto vector_a_to_by_in_line = vector_a_to_by / number_points_in_line;
    for (auto lm = 0; lm < number_points_in_line; lm++) {
      const auto m_x =
          min(heatmap_width - 1,
              positive_int_round_cuda(s_x + lm * vector_a_to_bx_in_line));
      const auto m_y =
          min(heatmap_height - 1,
              positive_int_round_cuda(s_y + lm * vector_a_to_by_in_line));
      const auto idx = m_y * heatmap_width + m_x;
      const auto score = (vector_a_to_b_norm_x * map_x[idx] +
                          vector_a_to_b_norm_y * map_y[idx]);
      if (score > inter_threshold) {
        sum += score;
        count++;
      }
    }
    // return paf score
    if (count / T(number_points_in_line) > inter_min_above_threshold)
      return sum / count;
    else {
      // ideally, if distance_ab = 0, paf is 0 between a and b, provoking a
      // false negative to fix it, we consider paf-connected keypoints very
      // close to have a minimum paf score, such that:
      //     1. it will consider very close keypoints (where the paf is 0)
      //     2. but it will not automatically connect them (case paf score = 1),
      //     or real paf might got
      //        missing
      const auto l2_dist = sqrtf(vector_a_to_bx * vector_a_to_bx +
                                 vector_a_to_by * vector_a_to_by);
      const auto threshold = sqrtf(heatmap_width * heatmap_height) /
                             150;  // 3.3 for 368x656, 6.6 for 2x resolution
      if (l2_dist < threshold)
        return T(
            default_nms_threshold +
            1e-6);  // without 1e-6 will not work because i use strict greater
    }
  }
  return -1;
}

template <typename T>
__global__ void paf_score_kernel(
    T* pair_scores_ptr, const T* const heat_map_ptr, const T* const peaks_ptr,
    const unsigned int* const body_part_pairs_ptr,
    const unsigned int* const map_idx_ptr, const unsigned int max_peaks,
    const int number_body_part_pairs, const int heatmap_width,
    const int heatmap_height, const T inter_threshold,
    const T inter_min_above_threshold, const T default_nms_threshold) {
  const auto peak_b = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto peak_a = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto pair_index = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (peak_a < max_peaks && peak_b < max_peaks)
  // if (pair_index < number_body_part_pairs && peak_a < max_peaks && peak_b <
  // max_peaks)
  {
    const auto base_index = 2 * pair_index;
    const auto part_a = body_part_pairs_ptr[base_index];
    const auto part_b = body_part_pairs_ptr[base_index + 1];

    const T number_peaks_a = peaks_ptr[3 * part_a * (max_peaks + 1)];
    const T number_peaks_b = peaks_ptr[3 * part_b * (max_peaks + 1)];

    const auto output_index =
        (pair_index * max_peaks + peak_a) * max_peaks + peak_b;
    if (peak_a < number_peaks_a && peak_b < number_peaks_b) {
      const auto map_idx_x = map_idx_ptr[base_index];
      const auto map_idx_y = map_idx_ptr[base_index + 1];

      const T* const body_part_a =
          peaks_ptr + (3 * (part_a * (max_peaks + 1) + peak_a + 1));
      const T* const body_part_b =
          peaks_ptr + (3 * (part_b * (max_peaks + 1) + peak_b + 1));
      const T* const map_x =
          heat_map_ptr + map_idx_x * heatmap_width * heatmap_height;
      const T* const map_y =
          heat_map_ptr + map_idx_y * heatmap_width * heatmap_height;
      pair_scores_ptr[output_index] = process(
          body_part_a, body_part_b, map_x, map_y, heatmap_width, heatmap_height,
          inter_threshold, inter_min_above_threshold, default_nms_threshold);
    } else
      pair_scores_ptr[output_index] = -1;
  }
}

template <typename T>
std::vector<std::tuple<T, T, int, int, int>> paf_ptr_into_vector_gpu(
    const T* const pair_scores, const T* const peaks_ptr, const int max_peaks,
    const std::vector<unsigned int>& body_part_pairs,
    const unsigned int number_body_part_pairs) {
  try {
    // result is a std::vector<std::tuple<double, double, int, int, int>> with:
    // (total_score, pa_fscore, pair_index, index_a, index_b)
    // total_score is first to simplify later sorting
    std::vector<std::tuple<T, T, int, int, int>> pair_connections;

    // get all paf pairs in a single std::vector
    const auto peaks_offset = 3 * (max_peaks + 1);
    for (auto pair_index = 0u; pair_index < number_body_part_pairs;
         pair_index++) {
      const auto body_part_a = body_part_pairs[2 * pair_index];
      const auto body_part_b = body_part_pairs[2 * pair_index + 1];
      const auto* candidate_a_ptr = peaks_ptr + body_part_a * peaks_offset;
      const auto* candidate_b_ptr = peaks_ptr + body_part_b * peaks_offset;
      const auto number_peaks_a = positive_int_round(candidate_a_ptr[0]);
      const auto number_peaks_b = positive_int_round(candidate_b_ptr[0]);
      const auto first_index = (int)pair_index * max_peaks * max_peaks;
      // e.g., neck-nose connection. for each neck
      for (auto index_a = 0; index_a < number_peaks_a; index_a++) {
        const auto i_index = first_index + index_a * max_peaks;
        // e.g., neck-nose connection. for each nose
        for (auto index_b = 0; index_b < number_peaks_b; index_b++) {
          const auto score_ab = pair_scores[i_index + index_b];

          // e.g., neck-nose connection. if possible paf between neck index_a,
          // nose index_b --> add parts score + connection score
          if (score_ab > 1e-6) {
            // total_score - only used for sorting
            // // original total_score
            // const auto total_score = score_ab;
            // improved total_score
            // improved to avoid too much weight in the paf between 2 elements,
            // adding some weight on their confidence (avoid connecting high
            // pa_fs on very low-confident keypoints)
            const auto index_score_a =
                body_part_a * peaks_offset + (index_a + 1) * 3 + 2;
            const auto index_score_b =
                body_part_b * peaks_offset + (index_b + 1) * 3 + 2;
            const auto total_score = score_ab +
                                     T(0.1) * peaks_ptr[index_score_a] +
                                     T(0.1) * peaks_ptr[index_score_b];
            // +1 because peaks_ptr starts with counter
            pair_connections.emplace_back(std::make_tuple(
                total_score, score_ab, pair_index, index_a + 1, index_b + 1));
          }
        }
      }
    }

    // sort rows in descending order based on its first element (`total_score`)
    if (!pair_connections.empty())
      std::sort(pair_connections.begin(), pair_connections.end(),
                std::greater<std::tuple<double, double, int, int, int>>());

    // return result
    return pair_connections;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return {};
  }
}

template <typename T>
void connect_body_parts_gpu(
    Array<T>& pose_keypoints, Array<T>& pose_scores,
    const T* const heat_map_gpu_ptr, const T* const peaks_ptr,
    const PoseModel pose_model, const Point<int>& heat_map_size,
    const int max_peaks, const T inter_min_above_threshold,
    const T inter_threshold, const int min_subset_cnt, const T min_subset_score,
    const T default_nms_threshold, const T scale_factor,
    const bool maximize_positives, T* pair_scores_gpu_ptr,
    const unsigned int* const body_part_pairs_gpu_ptr,
    const unsigned int* const map_idx_gpu_ptr, const T* const peaks_gpu_ptr) {
  try {
    // parts connection
    const auto& body_part_pairs = get_pose_part_pairs(pose_model);
    const auto number_body_parts = get_pose_number_body_parts(pose_model);
    const auto number_body_part_pairs =
        (unsigned int)(body_part_pairs.size() / 2);

    if (number_body_parts == 0)
      error("invalid value of number_body_parts, it must be positive, not " +
                std::to_string(number_body_parts),
            __LINE__, __FUNCTION__, __FILE__);
    if (body_part_pairs_gpu_ptr == nullptr || map_idx_gpu_ptr == nullptr)
      error(
          "the pointers body_part_pairs_gpu_ptr and map_idx_gpu_ptr cannot be "
          "nullptr.",
          __LINE__, __FUNCTION__, __FILE__);

    const dim3 threads_per_block{128, 1, 1};
    const dim3 num_blocks{
        get_number_cuda_blocks(max_peaks, threads_per_block.x),
        get_number_cuda_blocks(max_peaks, threads_per_block.y),
        get_number_cuda_blocks(number_body_part_pairs, threads_per_block.z)};
    paf_score_kernel<<<num_blocks, threads_per_block>>>(
        pair_scores_gpu_ptr, heat_map_gpu_ptr, peaks_gpu_ptr,
        body_part_pairs_gpu_ptr, map_idx_gpu_ptr, max_peaks,
        (int)number_body_part_pairs, heat_map_size.x, heat_map_size.y,
        inter_threshold, inter_min_above_threshold, default_nms_threshold);

    // get pair connections and their scores
    const auto pair_connections =
        paf_ptr_into_vector_gpu(pair_scores_gpu_ptr, peaks_ptr, max_peaks,
                                body_part_pairs, number_body_part_pairs);
    auto people_vector =
        paf_vector_into_people_vector(pair_connections, peaks_ptr, max_peaks,
                                      body_part_pairs, number_body_parts);
    // // old code: get pair connections and their scores
    // // std::vector<std::pair<std::vector<int>, double>> refers to:
    // //     - std::vector<int>: [body parts locations, #body parts found]
    // //     - double: person subset score
    // const T* const t_nullptr = nullptr;
    // const auto people_vector = create_people_vector(
    //     t_nullptr, peaks_ptr, pose_model, heat_map_size, max_peaks,
    //     inter_threshold, inter_min_above_threshold, body_part_pairs,
    //     number_body_parts, number_body_part_pairs, default_nms_threshold,
    //     pair_scores_cpu);
    // delete people below the following thresholds:
    // a) min_subset_cnt: removed if less than min_subset_cnt body parts
    // b) min_subset_score: removed if global score smaller than this
    // c) max_peaks (POSE_MAX_PEOPLE): keep first max_peaks people above
    // thresholds
    int number_people;
    std::vector<int> valid_subset_indexes;
    // valid_subset_indexes.reserve(fast_min((size_t)max_peaks,
    // people_vector.size()));
    valid_subset_indexes.reserve(people_vector.size());
    remove_people_below_thresholds_and_fill_faces(
        valid_subset_indexes, number_people, people_vector, number_body_parts,
        min_subset_cnt, min_subset_score, maximize_positives, peaks_ptr);
    // fill and return pose_keypoints
    people_vector_to_people_array(pose_keypoints, pose_scores, scale_factor,
                                  people_vector, valid_subset_indexes,
                                  peaks_ptr, number_people, number_body_parts,
                                  number_body_part_pairs);

    // sanity check
    cuda_check(__LINE__, __FUNCTION__, __FILE__);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template void connect_body_parts_gpu(
    Array<float>& pose_keypoints, Array<float>& pose_scores,
    const float* const heat_map_gpu_ptr, const float* const peaks_ptr,
    const PoseModel pose_model, const Point<int>& heat_map_size,
    const int max_peaks, const float inter_min_above_threshold,
    const float inter_threshold, const int min_subset_cnt,
    const float min_subset_score, const float scale_factor,
    const float default_nms_threshold, const bool maximize_positives,
    float* pair_scores_gpu_ptr,
    const unsigned int* const body_part_pairs_gpu_ptr,
    const unsigned int* const map_idx_gpu_ptr,
    const float* const peaks_gpu_ptr);

template void connect_body_parts_gpu(
    Array<double>& pose_keypoints, Array<double>& pose_scores,
    const double* const heat_map_gpu_ptr, const double* const peaks_ptr,
    const PoseModel pose_model, const Point<int>& heat_map_size,
    const int max_peaks, const double inter_min_above_threshold,
    const double inter_threshold, const int min_subset_cnt,
    const double min_subset_score, const double scale_factor,
    const double default_nms_threshold, const bool maximize_positives,
    double* pair_scores_gpu_ptr,
    const unsigned int* const body_part_pairs_gpu_ptr,
    const unsigned int* const map_idx_gpu_ptr,
    const double* const peaks_gpu_ptr);

}  // namespace openposert
