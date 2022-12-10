#pragma once

#include "half.hpp"

namespace openposert {

void paf_vector_into_people_vector(
    int* people_vector_body_ptr, half_float::half* people_vector_score_ptr,
    int* person_assigned_ptr, int* person_removed_ptr, int& people_vector_count,
    const int* const paf_sorted_ptr,
    const half_float::half* const paf_score_ptr,
    const int* const paf_pair_index_ptr, const int* const paf_index_a_ptr,
    const int* const paf_index_b_ptr, const int pair_connections_count,
    const half_float::half* const peaks_ptr, const int max_peaks,
    const unsigned int* const body_part_pairs,
    const unsigned int number_body_parts);

}
