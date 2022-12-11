#include "openposert/output/paf_vector_into_people_vector.hpp"

namespace openposert {

void paf_vector_into_people_vector(
    int* people_vector_body_ptr, float* people_vector_score_ptr,
    int* person_assigned_ptr, int* person_removed_ptr, int& people_vector_count,
    const int* const paf_sorted_ptr, const float* const paf_score_ptr,
    const int* const paf_pair_index_ptr, const int* const paf_index_a_ptr,
    const int* const paf_index_b_ptr, const int pair_connections_count,
    const float* const peaks_ptr, const int max_peaks,
    const unsigned int* const body_part_pairs,
    const unsigned int number_body_parts) {
  // std::vector<std::pair<std::vector<int>,
  // double>> refers to:
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
      const auto person_score =
          peaks_ptr[index_score_a] + peaks_ptr[index_score_b] + paf_score;
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
}

}  // namespace openposert
