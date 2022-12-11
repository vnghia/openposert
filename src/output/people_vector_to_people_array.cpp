#include "openposert/output/people_vector_to_people_array.hpp"

namespace openposert {

void people_vector_to_people_array(float* pose_keypoints, float* pose_scores,
                                   const float scale_factor,
                                   const int* const people_vector_body_ptr,
                                   const float* const people_vector_score_ptr,
                                   const int* const valid_subset_indexes_ptr,
                                   const int number_people,
                                   const float* const peaks_ptr,
                                   const unsigned int number_body_parts,
                                   const unsigned int number_body_part_pairs) {
  const auto vector_size = number_body_parts + 1;
  // fill people keypoints
  const auto one_over_number_body_parts_and_pa_fs =
      static_cast<float>(1.0f / (number_body_parts + number_body_part_pairs));
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
}

}  // namespace openposert
