#include "openposert/output/remove_people_below_thresholds_and_fill_faces.hpp"

namespace openposert {

void remove_people_below_thresholds_and_fill_faces(
    int* valid_subset_indexes_ptr, int& number_people,
    const int* const people_vector_body_ptr,
    const float* const people_vector_score_ptr, int* person_removed_ptr,
    const int people_vector_count, const unsigned int max_peaks,
    const unsigned int number_body_parts, const int min_subset_cnt,
    const float min_subset_score, const bool maximize_positives) {
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
}

}  // namespace openposert
