#pragma once

namespace openposert {

void people_vector_to_people_array(float* pose_keypoints, float* pose_scores,
                                   const float scale_factor,
                                   const int* const people_vector_body_ptr,
                                   const float* const people_vector_score_ptr,
                                   const int* const valid_subset_indexes_ptr,
                                   const int number_people,
                                   const float* const peaks_ptr,
                                   const unsigned int number_body_parts,
                                   const unsigned int number_body_part_pairs);

}
