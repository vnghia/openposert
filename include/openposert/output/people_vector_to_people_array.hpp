#pragma once

#include "half.hpp"

namespace openposert {

void people_vector_to_people_array(
    half_float::half* pose_keypoints, half_float::half* pose_scores,
    const half_float::half scale_factor,
    const int* const people_vector_body_ptr,
    const half_float::half* const people_vector_score_ptr,
    const int* const valid_subset_indexes_ptr, const int number_people,
    const half_float::half* const peaks_ptr,
    const unsigned int number_body_parts,
    const unsigned int number_body_part_pairs);

}
