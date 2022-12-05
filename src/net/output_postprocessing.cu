#include "minrt/utils.hpp"
#include "openposert/core/common.hpp"
#include "openposert/gpu/cuda.hpp"
#include "openposert/gpu/cuda_fast_math.hpp"
#include "openposert/net/output_postprocessing.hpp"
#include "openposert/utilities/fast_math.hpp"
#include "thrust/fill.h"
#include "thrust/sequence.h"
#include "thrust/sort.h"

namespace openposert {

using namespace minrt;

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

OutputPostprocessing::OutputPostprocessing(
    float* pose_keypoints, float* pose_scores, float* net_output_ptr,
    int net_output_width, int net_output_height, int peak_dim,
    float scale_factor, int number_body_parts, int number_body_part_pairs,
    int max_peaks, int min_subset_cnt, float min_subset_score,
    bool maximize_positives, float inter_threshold,
    float inter_min_above_threshold, float default_nms_threshold,
    float* peaks_ptr, float* pair_scores_ptr, unsigned int* body_part_pairs_ptr,
    unsigned int* pose_map_idx_ptr)
    : net_output_ptr(net_output_ptr),
      net_output_width(net_output_width),
      net_output_height(net_output_height),
      peak_dim(peak_dim),
      scale_factor(scale_factor),
      number_body_parts(number_body_parts),
      number_body_part_pairs(number_body_part_pairs),
      max_peaks(max_peaks),
      min_subset_cnt(min_subset_cnt),
      min_subset_score(min_subset_score),
      maximize_positives(maximize_positives),
      inter_threshold(inter_threshold),
      inter_min_above_threshold(inter_min_above_threshold),
      default_nms_threshold(default_nms_threshold),
      peaks_ptr(peaks_ptr),
      pair_scores_ptr(pair_scores_ptr),
      body_part_pairs_ptr(body_part_pairs_ptr),
      pose_map_idx_ptr(pose_map_idx_ptr),
      paf_total_size(number_body_part_pairs * max_peaks * max_peaks),
      pose_keypoints_(pose_keypoints),
      pose_scores_(pose_scores) {
  const int paf_total_size_int = paf_total_size * sizeof(int);
  const int paf_total_size_float = paf_total_size * sizeof(float);

  paf_sorted_index_ = cuda_malloc_managed(paf_total_size_int);
  paf_total_score_data_ = cuda_malloc_managed(paf_total_size_float);
  paf_score_data_ = cuda_malloc_managed(paf_total_size_float);
  paf_pair_index_data_ = cuda_malloc_managed(paf_total_size_int);
  paf_index_a_data_ = cuda_malloc_managed(paf_total_size_int);
  paf_index_b_data_ = cuda_malloc_managed(paf_total_size_int);

  const int people_vector_body_size =
      paf_total_size * (number_body_parts + 1) * sizeof(float);
  people_vector_body_data_ = cuda_malloc_managed(people_vector_body_size);
  people_vector_score_data_ = cuda_malloc_managed(paf_total_size_float);

  const int person_assigned_size = number_body_parts * max_peaks * sizeof(int);
  person_assigned_data_ = cuda_malloc_managed(person_assigned_size);
  person_removed_data_ = cuda_malloc_managed(paf_total_size_int);

  valid_subset_indexes_data_ = cuda_malloc_managed(paf_total_size_int);
}

int OutputPostprocessing::postprocessing_gpu() {
  const dim3 threads_per_block{128, 1, 1};
  const dim3 num_blocks{
      get_number_cuda_blocks(max_peaks, threads_per_block.x),
      get_number_cuda_blocks(max_peaks, threads_per_block.y),
      get_number_cuda_blocks(number_body_part_pairs, threads_per_block.z)};
  paf_score_kernel<<<num_blocks, threads_per_block>>>(
      pair_scores_ptr, net_output_ptr, peaks_ptr, body_part_pairs_ptr,
      pose_map_idx_ptr, max_peaks, number_body_part_pairs, net_output_width,
      net_output_height, inter_threshold, inter_min_above_threshold,
      default_nms_threshold);

  pair_connections_count_ = 0;
  thrust::sequence(thrust::host, static_cast<int*>(paf_sorted_index_.get()),
                   static_cast<int*>(paf_sorted_index_.get()) + paf_total_size,
                   0);
  paf_ptr_into_vector(static_cast<int*>(paf_sorted_index_.get()),
                      static_cast<float*>(paf_total_score_data_.get()),
                      static_cast<float*>(paf_score_data_.get()),
                      static_cast<int*>(paf_pair_index_data_.get()),
                      static_cast<int*>(paf_index_a_data_.get()),
                      static_cast<int*>(paf_index_b_data_.get()),
                      paf_total_size, pair_scores_ptr, peaks_ptr, max_peaks,
                      body_part_pairs_ptr, number_body_part_pairs,
                      pair_connections_count_);

  people_vector_count_ = 0;
  thrust::fill_n(thrust::host, static_cast<int*>(person_assigned_data_.get()),
                 number_body_parts * max_peaks, -1);
  thrust::fill_n(thrust::host,
                 static_cast<int*>(people_vector_body_data_.get()),
                 paf_total_size * (number_body_parts + 1), 0);
  paf_vector_into_people_vector(
      static_cast<int*>(people_vector_body_data_.get()),
      static_cast<float*>(people_vector_score_data_.get()),
      static_cast<int*>(person_assigned_data_.get()),
      static_cast<int*>(person_removed_data_.get()),
      static_cast<int*>(paf_sorted_index_.get()),
      static_cast<float*>(paf_score_data_.get()),
      static_cast<int*>(paf_pair_index_data_.get()),
      static_cast<int*>(paf_index_a_data_.get()),
      static_cast<int*>(paf_index_b_data_.get()), pair_connections_count_,
      peaks_ptr, max_peaks, body_part_pairs_ptr, number_body_parts,
      people_vector_count_);

  number_people_ = 0;
  remove_people_below_thresholds_and_fill_faces(
      static_cast<int*>(valid_subset_indexes_data_.get()), number_people_,
      static_cast<int*>(people_vector_body_data_.get()),
      static_cast<float*>(people_vector_score_data_.get()),
      static_cast<int*>(person_removed_data_.get()), people_vector_count_,
      number_body_parts, min_subset_cnt, min_subset_score, maximize_positives,
      peaks_ptr);

  thrust::fill_n(thrust::host, pose_keypoints_,
                 number_people_ * number_body_parts * peak_dim, 0);
  thrust::fill_n(thrust::host, pose_scores_, number_people_, 0);

  people_vector_to_people_array(
      pose_keypoints_, pose_scores_, scale_factor,
      static_cast<int*>(people_vector_body_data_.get()),
      static_cast<float*>(people_vector_score_data_.get()),
      static_cast<int*>(valid_subset_indexes_data_.get()), people_vector_count_,
      peaks_ptr, number_people_, number_body_parts, number_body_part_pairs);

  return number_people_;
}

void OutputPostprocessing::paf_ptr_into_vector(
    int* sorted_ptr, float* total_score_ptr, float* paf_score_ptr,
    int* pair_index_ptr, int* index_a_ptr, int* index_b_ptr,
    const int total_size, const float* const pair_scores,
    const float* const peaks_ptr, const int max_peaks,
    const unsigned int* body_part_pairs,
    const unsigned int number_body_part_pairs, int& pair_connections_count) {
  try {
    // result is a std::vector<std::tuple<double, double, int, int, int>> with:
    // (total_score, paf_score, pair_index, index_a, index_b)
    // total_score is first to simplify later sorting

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
                                     0.1f * peaks_ptr[index_score_a] +
                                     0.1f * peaks_ptr[index_score_b];
            // +1 because peaks_ptr starts with counter
            total_score_ptr[pair_connections_count] = total_score;
            paf_score_ptr[pair_connections_count] = score_ab;
            pair_index_ptr[pair_connections_count] = pair_index;
            index_a_ptr[pair_connections_count] = index_a + 1;
            index_b_ptr[pair_connections_count] = index_b + 1;
            ++pair_connections_count;
          }
        }
      }
    }
    if (pair_connections_count > 0) {
      thrust::sort_by_key(thrust::host, total_score_ptr,
                          total_score_ptr + pair_connections_count, sorted_ptr,
                          thrust::greater<float>());
    }

  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

void OutputPostprocessing::paf_vector_into_people_vector(
    int* people_vector_body_ptr, float* people_vector_score_ptr,
    int* person_assigned_ptr, int* person_removed_ptr,
    const int* const paf_sorted_ptr, const float* const paf_score_ptr,
    const int* const paf_pair_index_ptr, const int* const paf_index_a_ptr,
    const int* const paf_index_b_ptr, const int pair_connections_count,
    const float* const peaks_ptr, const int max_peaks,
    const unsigned int* body_part_pairs, const unsigned int number_body_parts,
    int& people_vector_count) {
  try {
    // std::vector<std::pair<std::vector<int>, double>> refers to:
    //     - std::vector<int>: [body parts locations, #body parts found]
    //     - double: person subset score
    const auto vector_size = number_body_parts + 1;
    const auto peaks_offset = (max_peaks + 1);
    // save which body parts have been already assigned
    // iterate over each paf pair connection detected
    // e.g., neck1-nose2, neck5-lshoulder0, etc.
    for (int i = 0; i < pair_connections_count; ++i) {
      // read pair_connection
      // // total score - only required for previous sort
      // const auto total_score = std::get<0>(pair_connection);
      const auto pair_connection = paf_sorted_ptr[i];
      const auto paf_score = paf_score_ptr[pair_connection];
      const auto pair_index = paf_pair_index_ptr[pair_connection];
      const auto index_a = paf_index_a_ptr[pair_connection];
      const auto index_b = paf_index_b_ptr[pair_connection];
      // derived data
      const auto body_part_a = body_part_pairs[2 * pair_index];
      const auto body_part_b = body_part_pairs[2 * pair_index + 1];

      const auto index_score_a = (body_part_a * peaks_offset + index_a) * 3 + 2;
      const auto index_score_b = (body_part_b * peaks_offset + index_b) * 3 + 2;
      // -1 because index_a and index_b are 1-based
      auto& a_assigned =
          person_assigned_ptr[body_part_a * max_peaks + index_a - 1];
      auto& b_assigned =
          person_assigned_ptr[body_part_b * max_peaks + index_b - 1];

      // different cases:
      //     1. a & b not assigned yet: create new person
      //     2. a assigned but not b: add b to person with a (if no another b
      //     there)
      //     3. b assigned but not a: add a to person with b (if no another a
      //     there)
      //     4. a & b already assigned to same person (circular/redundant paf):
      //     update person score
      //     5. a & b already assigned to different people: merge people if
      //     keypoint intersection is null
      // 1. a & b not assigned yet: create new person
      if (a_assigned < 0 && b_assigned < 0) {
        // keypoint indexes
        auto row_vector_ptr =
            people_vector_body_ptr + people_vector_count * vector_size;
        row_vector_ptr[body_part_a] = index_score_a;
        row_vector_ptr[body_part_b] = index_score_b;
        // number keypoints
        row_vector_ptr[vector_size - 1] = 2;
        // score
        const auto person_score = static_cast<float>(
            peaks_ptr[index_score_a] + peaks_ptr[index_score_b] + paf_score);
        // set associated person_assigned as assigned
        a_assigned = people_vector_count;
        b_assigned = a_assigned;
        // create new person_vector
        people_vector_score_ptr[people_vector_count] = person_score;
        ++people_vector_count;
      }
      // 2. a assigned but not b: add b to person with a (if no another b there)
      // or
      // 3. b assigned but not a: add a to person with b (if no another a there)
      else if ((a_assigned >= 0 && b_assigned < 0) ||
               (a_assigned < 0 && b_assigned >= 0)) {
        // assign person1 to one where x_assigned >= 0
        const auto assigned1 = (a_assigned >= 0 ? a_assigned : b_assigned);
        auto& assigned2 = (a_assigned >= 0 ? b_assigned : a_assigned);
        const auto body_part2 = (a_assigned >= 0 ? body_part_b : body_part_a);
        const auto index_score2 =
            (a_assigned >= 0 ? index_score_b : index_score_a);
        // person index
        auto person_vector = people_vector_body_ptr + assigned1 * vector_size;

        // if person with 1 does not have a 2 yet
        if (person_vector[body_part2] == 0) {
          // update keypoint indexes
          person_vector[body_part2] = index_score2;
          // update number keypoints
          ++person_vector[vector_size - 1];
          // update score
          people_vector_score_ptr[assigned1] +=
              peaks_ptr[index_score2] + paf_score;
          // set associated person_assigned as assigned
          assigned2 = assigned1;
        }
        // otherwise, ignore this b because the previous one came from a higher
        // paf-confident score
      }
      // 4. a & b already assigned to same person (circular/redundant paf):
      // update person score
      else if (a_assigned >= 0 && b_assigned >= 0 && a_assigned == b_assigned)
        people_vector_score_ptr[a_assigned] += paf_score;
      // 5. a & b already assigned to different people: merge people if keypoint
      // intersection is null i.e., that the keypoints in person a and b do not
      // overlap
      else if (a_assigned >= 0 && b_assigned >= 0 && a_assigned != b_assigned) {
        // assign person1 to the one with lowest index for 2 reasons:
        //     1. speed up: removing an element from std::vector is cheaper for
        //     latest elements
        //     2. avoid harder index update: updated elements in person1ssigned
        //     would depend on
        //        whether person1 > person2 or not: element = a_assigned -
        //        (person2 > person1 ? 1 : 0)
        const auto assigned1 =
            (a_assigned < b_assigned ? a_assigned : b_assigned);
        const auto assigned2 =
            (a_assigned < b_assigned ? b_assigned : a_assigned);
        auto person1 = people_vector_score_ptr + assigned1 * vector_size;
        const auto person2 = people_vector_score_ptr + assigned2 * vector_size;
        // check if complementary
        // defining found keypoint indexes in person_a as k_a, and analogously
        // k_b complementary if and only if k_a intersection k_b = empty. i.e.,
        // no common keypoints
        bool complementary = true;
        for (auto part = 0u; part < number_body_parts; part++) {
          if (person1[part] > 0 && person2[part] > 0) {
            complementary = false;
            break;
          }
        }
        // if complementary, merge both people into 1
        if (complementary) {
          // update keypoint indexes
          for (auto part = 0u; part < number_body_parts; part++)
            if (person1[part] == 0) person1[part] = person2[part];
          // update number keypoints
          person1[vector_size - 1] += person2[vector_size - 1];
          // update score
          people_vector_score_ptr[assigned1] +=
              people_vector_score_ptr[assigned2] + paf_score;
          // erase the non-merged person
          // people_vector.erase(people_vector.begin()+assigned2); // x2 slower
          // when removing on-the-fly
          person_removed_ptr[assigned2] = 1;
          // update associated person_assigned (person indexes have changed)
          for (int j = 0; j < number_body_parts * max_peaks; ++j) {
            if (person_assigned_ptr[j] == assigned2)
              person_assigned_ptr[j] = assigned1;
            // no need because i will only remove them at the very end
            // else if (element > assigned2)
            //     element--;
          }
        }
      }
    }
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

void OutputPostprocessing::remove_people_below_thresholds_and_fill_faces(
    int* valid_subset_indexes_ptr, int& number_people,
    int* people_vector_body_ptr, float* people_vector_score_ptr,
    int* person_removed_ptr, const int people_vector_count,
    const unsigned int number_body_parts, const int min_subset_cnt,
    const float min_subset_score, const bool maximize_positives,
    const float* const peaks_ptr) {
  try {
    const auto vector_size = number_body_parts + 1;

    const auto get_keypoint_counter =
        [people_vector_body_ptr, vector_size](
            int& person_counter, const unsigned int part, const int part_first,
            const int part_last, const int minimum) {
          // count keypoints
          int keypoint_counter = 0;
          for (auto i = part_first; i < part_last; i++)
            keypoint_counter +=
                (people_vector_body_ptr[part * vector_size + i] > 0);
          // if enough keypoints --> subtract them and keep them at least as big
          // as
          // minimum
          if (keypoint_counter > minimum)
            person_counter += minimum - keypoint_counter;
        };

    for (auto person = 0u; person < people_vector_count; ++person) {
      if (person_removed_ptr[person]) continue;

      auto person_counter =
          people_vector_body_ptr[(person + 1) * vector_size - 1];

      // foot keypoints do not affect person_counter (too many false positives,
      // same foot usually appears as both left and right keypoints)
      // pros: removed tons of false positives
      // cons: standalone leg will never be recorded
      // solution: no consider foot keypoints for that
      if (!maximize_positives &&
          (number_body_parts == 25 || number_body_parts > 70)) {
        const auto current_counter = person_counter;
        get_keypoint_counter(person_counter, person, 19, 25, 0);
        const auto new_counter = person_counter;
        // problem: same leg/foot keypoints are considered for both left and
        // right keypoints. solution: remove legs that are duplicated and that
        // do not have upper torso result: slight increase in coco m_ap and
        // decrease in m_ar + reducing a lot false positives!
        if (new_counter != current_counter && new_counter <= 4) continue;
      }
      // add only valid people
      const auto person_score = people_vector_score_ptr[person];
      if (person_counter >= min_subset_cnt &&
          (person_score / person_counter) >= min_subset_score) {
        valid_subset_indexes_ptr[number_people] = person;
        number_people++;
        // // this is not required, it is ok if there are more people. no more
        // gpu memory used. if (number_people == max_peaks)
        //     break;
        if (number_people == max_peaks) break;
      }
    }
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

void OutputPostprocessing::people_vector_to_people_array(
    float* pose_keypoints, float* pose_scores, const float scale_factor,
    const int* const people_vector_body_ptr,
    const float* const people_vector_score_ptr,
    const int* const valid_subset_indexes_ptr, const int people_vector_count,
    const float* const peaks_ptr, const int number_people,
    const unsigned int number_body_parts,
    const unsigned int number_body_part_pairs) {
  try {
    const auto vector_size = number_body_parts + 1;
    // fill people keypoints
    const auto one_over_number_body_parts_and_pa_fs =
        1 / static_cast<float>(number_body_parts + number_body_part_pairs);
    // for each person
    for (auto person = 0u; person < number_people; person++) {
      const auto person_idx = valid_subset_indexes_ptr[person];
      const auto person_vector =
          people_vector_body_ptr + vector_size * person_idx;
      // for each body part
      for (auto body_part = 0u; body_part < number_body_parts; body_part++) {
        const auto base_offset = (person * number_body_parts + body_part) * 3;
        const auto body_part_index = person_vector[body_part];
        if (body_part_index > 0) {
          pose_keypoints[base_offset] =
              peaks_ptr[body_part_index - 2] * scale_factor;
          pose_keypoints[base_offset + 1] =
              peaks_ptr[body_part_index - 1] * scale_factor;
          pose_keypoints[base_offset + 2] = peaks_ptr[body_part_index];
        }
      }
      pose_scores[person] = people_vector_score_ptr[person_idx] *
                            one_over_number_body_parts_and_pa_fs;
    }
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

}  // namespace openposert
