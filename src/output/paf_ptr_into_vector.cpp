#include "openposert/output/paf_ptr_into_vector.hpp"

#include "half.hpp"
#include "thrust/execution_policy.h"
#include "thrust/sort.h"

namespace openposert {

using namespace half_float::literal;

void paf_ptr_into_vector(int* sorted_ptr, half_float::half* total_score_ptr,
                         half_float::half* paf_score_ptr, int* pair_index_ptr,
                         int* index_a_ptr, int* index_b_ptr,
                         int& pair_connections_count,
                         const half_float::half* const pair_scores,
                         const half_float::half* const peaks_ptr,
                         const int max_peaks,
                         const unsigned int* const body_part_pairs,
                         const unsigned int number_body_part_pairs) {
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
    const auto number_peaks_a = half_float::half_cast<int>(candidate_a_ptr[0]);
    const auto number_peaks_b = half_float::half_cast<int>(candidate_b_ptr[0]);
    const auto first_index =
        static_cast<int>(pair_index) * max_peaks * max_peaks;

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
          const auto total_score = score_ab + 0.1_h * peaks_ptr[index_score_a] +
                                   0.1_h * peaks_ptr[index_score_b];
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
                        thrust::greater<half_float::half>());
  }
}

}  // namespace openposert
