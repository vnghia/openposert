#pragma once

namespace openposert {

void remove_people_below_thresholds_and_fill_faces(
    int* valid_subset_indexes_ptr, int& number_people,
    const int* const people_vector_body_ptr,
    const float* const people_vector_score_ptr, int* person_removed_ptr,
    const int people_vector_count, const unsigned int max_peaks,
    const unsigned int number_body_parts, const int min_subset_cnt,
    const float min_subset_score, const bool maximize_positives);

}
