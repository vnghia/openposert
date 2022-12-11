#pragma once

namespace openposert {

void paf_ptr_into_vector(int* sorted_ptr, float* total_score_ptr,
                         float* paf_score_ptr, int* pair_index_ptr,
                         int* index_a_ptr, int* index_b_ptr,
                         int& pair_connections_count,
                         const float* const pair_scores,
                         const float* const peaks_ptr, const int max_peaks,
                         const unsigned int* const body_part_pairs,
                         const unsigned int number_body_part_pairs);

}
