#include "openposert/net/body_part_connector.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

#include "openposert/core/array.hpp"
#include "openposert/core/common.hpp"
#include "openposert/core/point.hpp"
#include "openposert/core/rectangle.hpp"
#include "openposert/pose/enum.hpp"
#include "openposert/pose/pose_parameters.hpp"
#include "openposert/utilities/fast_math.hpp"
#include "openposert/utilities/keypoint.hpp"

namespace openposert {

template <typename T>
inline T get_score_ab(const int i, const int j, const T* const candidate_a_ptr,
                      const T* const candidate_b_ptr, const T* const map_x,
                      const T* const map_y, const Point<int>& heat_map_size,
                      const T inter_threshold,
                      const T inter_min_above_threshold,
                      const T default_nms_threshold) {
  try {
    const auto vector_a_to_bx = candidate_b_ptr[3 * j] - candidate_a_ptr[3 * i];
    const auto vector_a_to_by =
        candidate_b_ptr[3 * j + 1] - candidate_a_ptr[3 * i + 1];
    const auto vector_a_to_b_max =
        fast_max(std::abs(vector_a_to_bx), std::abs(vector_a_to_by));
    const auto number_points_in_line = fast_max(
        5, fast_min(25, positive_int_round(std::sqrt(5 * vector_a_to_b_max))));
    const auto vector_norm = T(std::sqrt(vector_a_to_bx * vector_a_to_bx +
                                         vector_a_to_by * vector_a_to_by));
    // if the peaks_ptr are coincident. don't connect them.
    if (vector_norm > 1e-6) {
      const auto s_x = candidate_a_ptr[3 * i];
      const auto s_y = candidate_a_ptr[3 * i + 1];
      const auto vector_a_to_b_norm_x = vector_a_to_bx / vector_norm;
      const auto vector_a_to_b_norm_y = vector_a_to_by / vector_norm;

      auto sum = T(0);
      auto count = 0u;
      const auto vector_a_to_bx_in_line =
          vector_a_to_bx / number_points_in_line;
      const auto vector_a_to_by_in_line =
          vector_a_to_by / number_points_in_line;
      for (auto lm = 0; lm < number_points_in_line; lm++) {
        const auto m_x = fast_max(
            0, fast_min(heat_map_size.x - 1,
                        positive_int_round(s_x + lm * vector_a_to_bx_in_line)));
        const auto m_y = fast_max(
            0, fast_min(heat_map_size.y - 1,
                        positive_int_round(s_y + lm * vector_a_to_by_in_line)));
        const auto idx = m_y * heat_map_size.x + m_x;
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
        //     2. but it will not automatically connect them (case paf score =
        //     1), or real paf might got
        //        missing
        const auto l2_dist = std::sqrt(vector_a_to_bx * vector_a_to_bx +
                                       vector_a_to_by * vector_a_to_by);
        const auto threshold =
            std::sqrt(heat_map_size.x * heat_map_size.y) / 150;
        if (l2_dist < threshold)
          return T(
              default_nms_threshold +
              1e-6);  // without 1e-6 will not work because i use strict greater
      }
    }
    return T(0);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return T(0);
  }
}

template <typename T>
void get_keypoint_counter(
    int& person_counter,
    const std::vector<std::pair<std::vector<int>, T>>& people_vector,
    const unsigned int part, const int part_first, const int part_last,
    const int minimum) {
  try {
    // count keypoints
    auto keypoint_counter = 0;
    for (auto i = part_first; i < part_last; i++)
      keypoint_counter += (people_vector[part].first.at(i) > 0);
    // if enough keypoints --> subtract them and keep them at least as big as
    // minimum
    if (keypoint_counter > minimum)
      person_counter +=
          minimum - keypoint_counter;  // person_counter = non-considered
                                       // keypoints + minimum
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
void get_roi_diameter_and_bounds(Rectangle<T>& roi, int& part_first_non0,
                                 int& part_last_non0,
                                 const std::vector<int>& person_vector,
                                 const T* const peaks_ptr, const int part_init,
                                 const int part_end, const T margin) {
  try {
    // find roi, part_first_non0, and part_last_non0
    roi = Rectangle<T>{std::numeric_limits<T>::max(),
                       std::numeric_limits<T>::max(), T(0), T(0)};
    part_first_non0 = -1;
    part_last_non0 = -1;
    for (auto part = part_init; part < part_end; part++) {
      const auto x = peaks_ptr[person_vector[part] - 2];
      const auto y = peaks_ptr[person_vector[part] - 1];
      const auto score = peaks_ptr[person_vector[part]];
      if (person_vector[part] > 0 && score > 0) {
        // roi
        if (roi.x > x) roi.x = x;
        if (roi.y > y) roi.y = y;
        if (roi.width < x) roi.width = x;
        if (roi.height < y) roi.height = y;
        // first keypoint?
        if (part_first_non0 < 0) part_first_non0 = part;
        // last keypoint?
        part_last_non0 = part;
      }
    }
    if (part_last_non0 > -1) {
      // add margin
      const auto margin_x = T(roi.width * margin);
      const auto margin_y = T(roi.height * margin);
      roi.x -= margin_x;
      roi.y -= margin_y;
      roi.width += 2 * margin_x;
      roi.height += 2 * margin_y;
      // part_first_non0+1 for loops
      part_last_non0++;
      // from [p1, p2] to [p1, width, height]
      // +1 to account for 1-line keypoints
      roi.width += 1 - roi.x;
      roi.height += 1 - roi.y;
    }
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
std::vector<std::pair<std::vector<int>, T>> create_people_vector(
    const T* const heat_map_ptr, const T* const peaks_ptr,
    const PoseModel pose_model, const Point<int>& heat_map_size,
    const int max_peaks, const T inter_threshold,
    const T inter_min_above_threshold,
    const std::vector<unsigned int>& body_part_pairs,
    const unsigned int number_body_parts,
    const unsigned int number_body_part_pairs, const T default_nms_threshold,
    const Array<T>& pair_scores) {
  try {
    // std::vector<std::pair<std::vector<int>, double>> refers to:
    //     - std::vector<int>: [body parts locations, #body parts found]
    //     - double: person subset score
    std::vector<std::pair<std::vector<int>, T>> people_vector;
    const auto& map_idx = get_pose_map_index(pose_model);
    const auto number_body_parts_and_bkg =
        number_body_parts + (add_bkg_channel(pose_model) ? 1 : 0);
    const auto vector_size = number_body_parts + 1;
    const auto peaks_offset = 3 * (max_peaks + 1);
    const auto heat_map_offset = heat_map_size.area();
    // iterate over it paf connection, e.g., neck-nose, neck-lshoulder, etc.
    for (auto pair_index = 0u; pair_index < number_body_part_pairs;
         pair_index++) {
      const auto body_part_a = body_part_pairs[2 * pair_index];
      const auto body_part_b = body_part_pairs[2 * pair_index + 1];
      const auto* candidate_a_ptr = peaks_ptr + body_part_a * peaks_offset;
      const auto* candidate_b_ptr = peaks_ptr + body_part_b * peaks_offset;
      const auto number_peaks_a = positive_int_round(candidate_a_ptr[0]);
      const auto number_peaks_b = positive_int_round(candidate_b_ptr[0]);

      // e.g., neck-nose connection. if one of them is empty (e.g., no noses
      // detected) add the non-empty elements into the people_vector
      if (number_peaks_a == 0 || number_peaks_b == 0) {
        // e.g., neck-nose connection. if no necks, add all noses
        // change w.r.t. other
        if (number_peaks_a == 0)  // number_peaks_b == 0 or not
        {
          // non-mpi
          if (number_body_parts != 15) {
            for (auto i = 1; i <= number_peaks_b; i++) {
              bool found = false;
              for (const auto& person_vector : people_vector) {
                const auto off = (int)body_part_b * peaks_offset + i * 3 + 2;
                if (person_vector.first[body_part_b] == off) {
                  found = true;
                  break;
                }
              }
              // add new person_vector with this element
              if (!found) {
                std::vector<int> row_vector(vector_size, 0);
                // store the index
                row_vector[body_part_b] =
                    body_part_b * peaks_offset + i * 3 + 2;
                // last number in each row is the parts number of that person
                row_vector.back() = 1;
                const auto person_score = candidate_b_ptr[i * 3 + 2];
                // second last number in each row is the total score
                people_vector.emplace_back(
                    std::make_pair(row_vector, person_score));
              }
            }
          }
          // mpi
          else {
            for (auto i = 1; i <= number_peaks_b; i++) {
              std::vector<int> row_vector(vector_size, 0);
              // store the index
              row_vector[body_part_b] = body_part_b * peaks_offset + i * 3 + 2;
              // last number in each row is the parts number of that person
              row_vector.back() = 1;
              // second last number in each row is the total score
              const auto person_score = candidate_b_ptr[i * 3 + 2];
              people_vector.emplace_back(
                  std::make_pair(row_vector, person_score));
            }
          }
        }
        // e.g., neck-nose connection. if no noses, add all necks
        else  // if (number_peaks_a != 0 && number_peaks_b == 0)
        {
          // non-mpi
          if (number_body_parts != 15) {
            for (auto i = 1; i <= number_peaks_a; i++) {
              bool found = false;
              const auto index_a = body_part_a;
              for (const auto& person_vector : people_vector) {
                const auto off = (int)body_part_a * peaks_offset + i * 3 + 2;
                if (person_vector.first[index_a] == off) {
                  found = true;
                  break;
                }
              }
              if (!found) {
                std::vector<int> row_vector(vector_size, 0);
                // store the index
                row_vector[body_part_a] =
                    body_part_a * peaks_offset + i * 3 + 2;
                // last number in each row is the parts number of that person
                row_vector.back() = 1;
                // second last number in each row is the total score
                const auto person_score = candidate_a_ptr[i * 3 + 2];
                people_vector.emplace_back(
                    std::make_pair(row_vector, person_score));
              }
            }
          }
          // mpi
          else {
            for (auto i = 1; i <= number_peaks_a; i++) {
              std::vector<int> row_vector(vector_size, 0);
              // store the index
              row_vector[body_part_a] = body_part_a * peaks_offset + i * 3 + 2;
              // last number in each row is the parts number of that person
              row_vector.back() = 1;
              // second last number in each row is the total score
              const auto person_score = candidate_a_ptr[i * 3 + 2];
              people_vector.emplace_back(
                  std::make_pair(row_vector, person_score));
            }
          }
        }
      }
      // e.g., neck-nose connection. if necks and noses, look for maximums
      else  // if (number_peaks_a != 0 && number_peaks_b != 0)
      {
        // (score, index_a, index_b). inverted order for easy std::sort
        std::vector<std::tuple<double, int, int>> all_ab_connections;
        // note: problem of this function, if no right paf between a and b, both
        // elements are discarded. however, they should be added independently,
        // not discarded
        if (heat_map_ptr != nullptr) {
          const auto* map_x = heat_map_ptr + (number_body_parts_and_bkg +
                                              map_idx[2 * pair_index]) *
                                                 heat_map_offset;
          const auto* map_y = heat_map_ptr + (number_body_parts_and_bkg +
                                              map_idx[2 * pair_index + 1]) *
                                                 heat_map_offset;
          // e.g., neck-nose connection. for each neck
          for (auto i = 1; i <= number_peaks_a; i++) {
            // e.g., neck-nose connection. for each nose
            for (auto j = 1; j <= number_peaks_b; j++) {
              // initial paf
              auto score_ab = get_score_ab(
                  i, j, candidate_a_ptr, candidate_b_ptr, map_x, map_y,
                  heat_map_size, inter_threshold, inter_min_above_threshold,
                  default_nms_threshold);

              // e.g., neck-nose connection. if possible paf between neck i,
              // nose j --> add parts score + connection score
              if (score_ab > 1e-6)
                all_ab_connections.emplace_back(
                    std::make_tuple(score_ab, i, j));
            }
          }
        } else if (!pair_scores.empty()) {
          const auto first_index = (int)pair_index * pair_scores.get_size(1) *
                                   pair_scores.get_size(2);
          // e.g., neck-nose connection. for each neck
          for (auto i = 0; i < number_peaks_a; i++) {
            const auto i_index = first_index + i * pair_scores.get_size(2);
            // e.g., neck-nose connection. for each nose
            for (auto j = 0; j < number_peaks_b; j++) {
              const auto score_ab = pair_scores[i_index + j];

              // e.g., neck-nose connection. if possible paf between neck i,
              // nose j --> add parts score + connection score
              if (score_ab > 1e-6)
                // +1 because peaks_ptr starts with counter
                all_ab_connections.emplace_back(
                    std::make_tuple(score_ab, i + 1, j + 1));
            }
          }
        } else
          error("error. should not reach here.", __LINE__, __FUNCTION__,
                __FILE__);

        // select the top min_ab connection, assuming that each part occur only
        // once sort rows in descending order based on parts + connection score
        if (!all_ab_connections.empty())
          std::sort(all_ab_connections.begin(), all_ab_connections.end(),
                    std::greater<std::tuple<double, int, int>>());

        std::vector<std::tuple<int, int, double>>
            ab_connections;  // (x, y, score)
        {
          const auto min_ab = fast_min(number_peaks_a, number_peaks_b);
          std::vector<int> occur_a(number_peaks_a, 0);
          std::vector<int> occur_b(number_peaks_b, 0);
          auto counter = 0;
          for (const auto& a_b_connection : all_ab_connections) {
            const auto score = std::get<0>(a_b_connection);
            const auto index_a = std::get<1>(a_b_connection);
            const auto index_b = std::get<2>(a_b_connection);
            if (!occur_a[index_a - 1] && !occur_b[index_b - 1]) {
              ab_connections.emplace_back(std::make_tuple(
                  body_part_a * peaks_offset + index_a * 3 + 2,
                  body_part_b * peaks_offset + index_b * 3 + 2, score));
              counter++;
              if (counter == min_ab) break;
              occur_a[index_a - 1] = 1;
              occur_b[index_b - 1] = 1;
            }
          }
        }

        // cluster all the body part candidates into people_vector based on the
        // part connection
        if (!ab_connections.empty()) {
          // initialize first body part connection 15&16
          if (pair_index == 0) {
            for (const auto& ab_connection : ab_connections) {
              std::vector<int> row_vector(number_body_parts + 3, 0);
              const auto index_a = std::get<0>(ab_connection);
              const auto index_b = std::get<1>(ab_connection);
              const auto score = std::get<2>(ab_connection);
              row_vector[body_part_pairs[0]] = index_a;
              row_vector[body_part_pairs[1]] = index_b;
              row_vector.back() = 2;
              // add the score of parts and the connection
              const auto person_score =
                  T(peaks_ptr[index_a] + peaks_ptr[index_b] + score);
              people_vector.emplace_back(
                  std::make_pair(row_vector, person_score));
            }
          }
          // add ears connections (in case person is looking to opposite
          // direction to camera) note: this has some issues:
          //     - it does not prevent repeating the same keypoint in different
          //     people
          //     - assuming i have nose,eye,ear as 1 person subset, and whole
          //     arm as another one, it
          //       will not merge them both
          else if ((number_body_parts == 18 &&
                    (pair_index == 17 || pair_index == 18)) ||
                   ((number_body_parts == 19 || (number_body_parts == 25) ||
                     number_body_parts == 59 || number_body_parts == 65) &&
                    (pair_index == 18 || pair_index == 19))) {
            for (const auto& ab_connection : ab_connections) {
              const auto index_a = std::get<0>(ab_connection);
              const auto index_b = std::get<1>(ab_connection);
              for (auto& person_vector : people_vector) {
                auto& person_vector_a = person_vector.first[body_part_a];
                auto& person_vector_b = person_vector.first[body_part_b];
                if (person_vector_a == index_a && person_vector_b == 0) {
                  person_vector_b = index_b;
                  // // this seems to harm acc 0.1% for body_25
                  // person_vector.first.back()++;
                } else if (person_vector_b == index_b && person_vector_a == 0) {
                  person_vector_a = index_a;
                  // // this seems to harm acc 0.1% for body_25
                  // person_vector.first.back()++;
                }
              }
            }
          } else {
            // a is already in the people_vector, find its connection b
            for (const auto& ab_connection : ab_connections) {
              const auto index_a = std::get<0>(ab_connection);
              const auto index_b = std::get<1>(ab_connection);
              const auto score = T(std::get<2>(ab_connection));
              bool found = false;
              for (auto& person_vector : people_vector) {
                // found part_a in a people_vector, add part_b to same one.
                if (person_vector.first[body_part_a] == index_a) {
                  person_vector.first[body_part_b] = index_b;
                  person_vector.first.back()++;
                  person_vector.second += peaks_ptr[index_b] + score;
                  found = true;
                  break;
                }
              }
              // not found part_a in people_vector, add new people_vector
              // element
              if (!found) {
                std::vector<int> row_vector(vector_size, 0);
                row_vector[body_part_a] = index_a;
                row_vector[body_part_b] = index_b;
                row_vector.back() = 2;
                const auto person_score =
                    T(peaks_ptr[index_a] + peaks_ptr[index_b] + score);
                people_vector.emplace_back(
                    std::make_pair(row_vector, person_score));
              }
            }
          }
        }
      }
    }
    return people_vector;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return {};
  }
}

template <typename T>
std::vector<std::tuple<T, T, int, int, int>> paf_ptr_into_vector(
    const Array<T>& pair_scores, const T* const peaks_ptr, const int max_peaks,
    const std::vector<unsigned int>& body_part_pairs,
    const unsigned int number_body_part_pairs) {
  try {
    // result is a std::vector<std::tuple<double, double, int, int, int>> with:
    // (total_score, pa_fscore, pair_index, index_a, index_b)
    // total_score is first to simplify later sorting
    std::vector<std::tuple<T, T, int, int, int>> pair_connections;

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
      const auto first_index =
          (int)pair_index * pair_scores.get_size(1) * pair_scores.get_size(2);
      // e.g., neck-nose connection. for each neck
      for (auto index_a = 0; index_a < number_peaks_a; index_a++) {
        const auto i_index = first_index + index_a * pair_scores.get_size(2);
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
                                     T(0.1) * peaks_ptr[index_score_a] +
                                     T(0.1) * peaks_ptr[index_score_b];
            // +1 because peaks_ptr starts with counter
            pair_connections.emplace_back(std::make_tuple(
                total_score, score_ab, pair_index, index_a + 1, index_b + 1));
          }
        }
      }
    }

    // sort rows in descending order based on its first element (`total_score`)
    if (!pair_connections.empty())
      std::sort(pair_connections.begin(), pair_connections.end(),
                std::greater<std::tuple<double, double, int, int, int>>());

    // return result
    return pair_connections;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return {};
  }
}

template <typename T>
std::vector<std::pair<std::vector<int>, T>> paf_vector_into_people_vector(
    const std::vector<std::tuple<T, T, int, int, int>>& pair_connections,
    const T* const peaks_ptr, const int max_peaks,
    const std::vector<unsigned int>& body_part_pairs,
    const unsigned int number_body_parts) {
  try {
    // std::vector<std::pair<std::vector<int>, double>> refers to:
    //     - std::vector<int>: [body parts locations, #body parts found]
    //     - double: person subset score
    std::vector<std::pair<std::vector<int>, T>> people_vector;
    const auto vector_size = number_body_parts + 1;
    const auto peaks_offset = (max_peaks + 1);
    // save which body parts have been already assigned
    std::vector<int> person_assigned(number_body_parts * max_peaks, -1);
    std::set<int, std::greater<int>> indexes_to_remove_sorted_set;
    // iterate over each paf pair connection detected
    // e.g., neck1-nose2, neck5-lshoulder0, etc.
    for (const auto& pair_connection : pair_connections) {
      // read pair_connection
      // // total score - only required for previous sort
      // const auto total_score = std::get<0>(pair_connection);
      const auto paf_score = std::get<1>(pair_connection);
      const auto pair_index = std::get<2>(pair_connection);
      const auto index_a = std::get<3>(pair_connection);
      const auto index_b = std::get<4>(pair_connection);
      // derived data
      const auto body_part_a = body_part_pairs[2 * pair_index];
      const auto body_part_b = body_part_pairs[2 * pair_index + 1];

      const auto index_score_a = (body_part_a * peaks_offset + index_a) * 3 + 2;
      const auto index_score_b = (body_part_b * peaks_offset + index_b) * 3 + 2;
      // -1 because index_a and index_b are 1-based
      auto& a_assigned = person_assigned[body_part_a * max_peaks + index_a - 1];
      auto& b_assigned = person_assigned[body_part_b * max_peaks + index_b - 1];

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
        std::vector<int> row_vector(vector_size, 0);
        row_vector[body_part_a] = index_score_a;
        row_vector[body_part_b] = index_score_b;
        // number keypoints
        row_vector.back() = 2;
        // score
        const auto person_score =
            T(peaks_ptr[index_score_a] + peaks_ptr[index_score_b] + paf_score);
        // set associated person_assigned as assigned
        a_assigned = (int)people_vector.size();
        b_assigned = a_assigned;
        // create new person_vector
        people_vector.emplace_back(std::make_pair(row_vector, person_score));
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
        auto& person_vector = people_vector[assigned1];

        // if person with 1 does not have a 2 yet
        if (person_vector.first[body_part2] == 0) {
          // update keypoint indexes
          person_vector.first[body_part2] = index_score2;
          // update number keypoints
          person_vector.first.back()++;
          // update score
          person_vector.second += peaks_ptr[index_score2] + paf_score;
          // set associated person_assigned as assigned
          assigned2 = assigned1;
        }
        // otherwise, ignore this b because the previous one came from a higher
        // paf-confident score
      }
      // 4. a & b already assigned to same person (circular/redundant paf):
      // update person score
      else if (a_assigned >= 0 && b_assigned >= 0 && a_assigned == b_assigned)
        people_vector[a_assigned].second += paf_score;
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
        auto& person1 = people_vector[assigned1].first;
        const auto& person2 = people_vector[assigned2].first;
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
          person1.back() += person2.back();
          // update score
          people_vector[assigned1].second +=
              people_vector[assigned2].second + paf_score;
          // erase the non-merged person
          // people_vector.erase(people_vector.begin()+assigned2); // x2 slower
          // when removing on-the-fly
          indexes_to_remove_sorted_set.emplace(
              assigned2);  // add into set so we can remove them all at once
          // update associated person_assigned (person indexes have changed)
          for (auto& element : person_assigned) {
            if (element == assigned2) element = assigned1;
            // no need because i will only remove them at the very end
            // else if (element > assigned2)
            //     element--;
          }
        }
      }
    }
    // remove unused people
    for (const auto& index : indexes_to_remove_sorted_set)
      people_vector.erase(people_vector.begin() + index);
    // return result
    return people_vector;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return {};
  }
}

template <typename T>
void remove_people_below_thresholds_and_fill_faces(
    std::vector<int>& valid_subset_indexes, int& number_people,
    std::vector<std::pair<std::vector<int>, T>>& people_vector,
    const unsigned int number_body_parts, const int min_subset_cnt,
    const T min_subset_score, const bool maximize_positives,
    const T* const peaks_ptr)
// const int min_subset_cnt, const T min_subset_score, const int max_peaks,
// const bool maximize_positives)
{
  try {
    // delete people below the following thresholds:
    // a) min_subset_cnt: removed if less than min_subset_cnt body parts
    // b) min_subset_score: removed if global score smaller than this
    // c) max_peaks (POSE_MAX_PEOPLE): keep first max_peaks people above
    // thresholds -> not required
    number_people = 0;
    valid_subset_indexes.clear();
    // valid_subset_indexes.reserve(fast_min((size_t)max_peaks,
    // people_vector.size())); // max_peaks is not required
    valid_subset_indexes.reserve(people_vector.size());
    // face valid sets
    std::vector<int> face_valid_subset_indexes;
    face_valid_subset_indexes.reserve(people_vector.size());
    // face invalid sets
    std::vector<int> face_invalid_subset_indexes;
    if (number_body_parts >= 135)
      face_invalid_subset_indexes.reserve(people_vector.size());
    // for each person candidate
    for (auto person = 0u; person < people_vector.size(); person++) {
      auto person_counter = people_vector[person].first.back();
      // analog for hand/face keypoints
      if (number_body_parts >= 135) {
        // no consider face keypoints for person_counter
        const auto current_counter = person_counter;
        get_keypoint_counter(person_counter, people_vector, person, 65, 135, 1);
        const auto new_counter = person_counter;
        if (person_counter == 1) {
          face_invalid_subset_indexes.emplace_back(person);
          continue;
        }
        // if body is still valid and facial points were removed, then add to
        // valid faces
        else if (current_counter != new_counter)
          face_valid_subset_indexes.emplace_back(person);
        // no consider right hand keypoints for person_counter
        get_keypoint_counter(person_counter, people_vector, person, 45, 65, 1);
        // no consider left hand keypoints for person_counter
        get_keypoint_counter(person_counter, people_vector, person, 25, 45, 1);
      }
      // foot keypoints do not affect person_counter (too many false positives,
      // same foot usually appears as both left and right keypoints)
      // pros: removed tons of false positives
      // cons: standalone leg will never be recorded
      // solution: no consider foot keypoints for that
      if (!maximize_positives &&
          (number_body_parts == 25 || number_body_parts > 70)) {
        const auto current_counter = person_counter;
        get_keypoint_counter(person_counter, people_vector, person, 19, 25, 0);
        const auto new_counter = person_counter;
        // problem: same leg/foot keypoints are considered for both left and
        // right keypoints. solution: remove legs that are duplicated and that
        // do not have upper torso result: slight increase in coco m_ap and
        // decrease in m_ar + reducing a lot false positives!
        if (new_counter != current_counter && new_counter <= 4) continue;
      }
      // add only valid people
      const auto person_score = people_vector[person].second;
      if (person_counter >= min_subset_cnt &&
          (person_score / person_counter) >= min_subset_score) {
        number_people++;
        valid_subset_indexes.emplace_back(person);
        // // this is not required, it is ok if there are more people. no more
        // gpu memory used. if (number_people == max_peaks)
        //     break;
      }
      // sanity check
      else if ((person_counter < 1 && number_body_parts != 25 &&
                number_body_parts < 70) ||
               person_counter < 0)
        error("bad person_counter (" + std::to_string(person_counter) +
                  "). bug in this"
                  " function if this happens.",
              __LINE__, __FUNCTION__, __FILE__);
    }
    // random standalone facial keypoints --> merge into a more complete face
    if (number_people > 0) {
      // check invalid faces
      for (const auto& person_invalid : face_invalid_subset_indexes) {
        // get roi of current face
        Rectangle<T> roi_invalid;
        int part_first_non0_invalid = -1;
        int part_last_non0_invalid = -1;
        get_roi_diameter_and_bounds(
            roi_invalid, part_first_non0_invalid, part_last_non0_invalid,
            people_vector[person_invalid].first, peaks_ptr, 65, 135, T(0.2));
        // check all valid faces to find best candidate
        float keypoints_roi_best = 0.f;
        auto keypoints_roi_best_index = -1;
        for (auto person_id = 0u; person_id < face_valid_subset_indexes.size();
             person_id++) {
          const auto& person_valid = face_valid_subset_indexes[person_id];
          // get roi of current face
          Rectangle<T> roi_valid;
          int part_first_non0_valid = -1;
          int part_last_non0_valid = -1;
          get_roi_diameter_and_bounds(
              roi_valid, part_first_non0_valid, part_last_non0_valid,
              people_vector[person_valid].first, peaks_ptr, 65, 135, T(0.1));
          // get roi between both faces
          const auto keypoints_roi = get_keypoints_roi(roi_valid, roi_invalid);
          // update best so far
          if (keypoints_roi_best < keypoints_roi) {
            keypoints_roi_best = keypoints_roi;
            keypoints_roi_best_index = person_id;
          }
        }
        // if invalid and best valid candidate overlap enough --> merge them
        if (keypoints_roi_best > 0.3f ||
            (keypoints_roi_best > 0.01f &&
             face_valid_subset_indexes.size() < 3)) {
          const auto& person_valid =
              face_valid_subset_indexes[keypoints_roi_best_index];
          // if it is from that face --> combine invalid face keypoints into
          // valid face
          for (auto part = part_first_non0_invalid;
               part < part_last_non0_invalid; part++) {
            auto& person_vector_valid = people_vector[person_valid].first;
            const auto score_valid = peaks_ptr[person_vector_valid[part]];
            const auto& person_vector_invalid =
                people_vector[person_invalid].first;
            const auto score_invalid = peaks_ptr[person_vector_invalid[part]];
            // if the new one has a keypoint...
            if (person_vector_invalid[part] != 0) {
              // ... and the original face does not have it, then add it to it
              if (person_vector_valid[part] == 0) {
                if (person_vector_invalid[part] != 0) {
                  person_vector_valid[part] = person_vector_invalid[part];
                  people_vector[person_valid].second += score_invalid;
                }
              }
              // ... and its score is higher than the original one, then replace
              // it
              else if (score_valid < score_invalid) {
                person_vector_valid[part] = person_vector_invalid[part];
                people_vector[person_valid].second +=
                    score_invalid - score_valid;
              }
            }
          }
        }
      }
    }
    // if no people found --> repeat with maximize_positives = true
    // result: increased coco m_ap because we catch more foot-only images
    if (number_people == 0 && !maximize_positives) {
      remove_people_below_thresholds_and_fill_faces(
          valid_subset_indexes, number_people, people_vector, number_body_parts,
          min_subset_cnt, min_subset_score, true, peaks_ptr);
      // // debugging
      // if (number_people > 0)
      //     op_log("found " + std::to_string(number_people) + " people in
      //     second iteration");
    }
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
void people_vector_to_people_array(
    Array<T>& pose_keypoints, Array<T>& pose_scores, const T scale_factor,
    const std::vector<std::pair<std::vector<int>, T>>& people_vector,
    const std::vector<int>& valid_subset_indexes, const T* const peaks_ptr,
    const int number_people, const unsigned int number_body_parts,
    const unsigned int number_body_part_pairs) {
  try {
    // allocate memory (initialized to 0)
    if (number_people > 0) {
      // initialized to 0 for non-found keypoints in people
      pose_keypoints.reset({number_people, (int)number_body_parts, 3}, 0.f);
      pose_scores.reset(number_people);
    }
    // no people --> empty arrays
    else {
      pose_keypoints.reset();
      pose_scores.reset();
    }
    // fill people keypoints
    const auto one_over_number_body_parts_and_pa_fs =
        1 / T(number_body_parts + number_body_part_pairs);
    // for each person
    for (auto person = 0u; person < valid_subset_indexes.size(); person++) {
      const auto& person_pair = people_vector[valid_subset_indexes[person]];
      const auto& person_vector = person_pair.first;
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
      pose_scores[person] =
          person_pair.second * one_over_number_body_parts_and_pa_fs;
    }
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
void connect_body_parts_cpu(
    Array<T>& pose_keypoints, Array<T>& pose_scores,
    const T* const heat_map_ptr, const T* const peaks_ptr,
    const PoseModel pose_model, const Point<int>& heat_map_size,
    const int max_peaks, const T inter_min_above_threshold,
    const T inter_threshold, const int min_subset_cnt, const T min_subset_score,
    const T default_nms_threshold, const T scale_factor,
    const bool maximize_positives) {
  try {
    // parts connection
    const auto& body_part_pairs = get_pose_part_pairs(pose_model);
    const auto number_body_parts = get_pose_number_body_parts(pose_model);
    const auto number_body_part_pairs =
        (unsigned int)(body_part_pairs.size() / 2);
    if (number_body_parts == 0)
      error("invalid value of number_body_parts, it must be positive, not " +
                std::to_string(number_body_parts),
            __LINE__, __FUNCTION__, __FILE__);
    // std::vector<std::pair<std::vector<int>, double>> refers to:
    //     - std::vector<int>: [body parts locations, #body parts found]
    //     - double: person subset score

    auto people_vector = create_people_vector(
        heat_map_ptr, peaks_ptr, pose_model, heat_map_size, max_peaks - 1,
        inter_threshold, inter_min_above_threshold, body_part_pairs,
        number_body_parts, number_body_part_pairs, default_nms_threshold);
    // delete people below the following thresholds:
    // a) min_subset_cnt: removed if less than min_subset_cnt body parts
    // b) min_subset_score: removed if global score smaller than this
    // c) max_peaks (pose_max_people): keep first max_peaks people above
    // thresholds
    int number_people;
    std::vector<int> valid_subset_indexes;
    // valid_subset_indexes.reserve(fast_min((size_t)max_peaks,
    // people_vector.size()));
    valid_subset_indexes.reserve(people_vector.size());
    remove_people_below_thresholds_and_fill_faces(
        valid_subset_indexes, number_people, people_vector, number_body_parts,
        min_subset_cnt, min_subset_score, maximize_positives, peaks_ptr);
    // fill and return pose_keypoints
    people_vector_to_people_array(pose_keypoints, pose_scores, scale_factor,
                                  people_vector, valid_subset_indexes,
                                  peaks_ptr, number_people, number_body_parts,
                                  number_body_part_pairs);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template void connect_body_parts_cpu(
    Array<float>& pose_keypoints, Array<float>& pose_scores,
    const float* const heat_map_ptr, const float* const peaks_ptr,
    const PoseModel pose_model, const Point<int>& heat_map_size,
    const int max_peaks, const float inter_min_above_threshold,
    const float inter_threshold, const int min_subset_cnt,
    const float min_subset_score, const float default_nms_threshold,
    const float scale_factor, const bool maximize_positives);
template void connect_body_parts_cpu(
    Array<double>& pose_keypoints, Array<double>& pose_scores,
    const double* const heat_map_ptr, const double* const peaks_ptr,
    const PoseModel pose_model, const Point<int>& heat_map_size,
    const int max_peaks, const double inter_min_above_threshold,
    const double inter_threshold, const int min_subset_cnt,
    const double min_subset_score, const double default_nms_threshold,
    const double scale_factor, const bool maximize_positives);

template std::vector<std::pair<std::vector<int>, float>> create_people_vector(
    const float* const heat_map_ptr, const float* const peaks_ptr,
    const PoseModel pose_model, const Point<int>& heat_map_size,
    const int max_peaks, const float inter_threshold,
    const float inter_min_above_threshold,
    const std::vector<unsigned int>& body_part_pairs,
    const unsigned int number_body_parts,
    const unsigned int number_body_part_pairs,
    const float default_nms_threshold, const Array<float>& precomputed_pa_fs);
template std::vector<std::pair<std::vector<int>, double>> create_people_vector(
    const double* const heat_map_ptr, const double* const peaks_ptr,
    const PoseModel pose_model, const Point<int>& heat_map_size,
    const int max_peaks, const double inter_threshold,
    const double inter_min_above_threshold,
    const std::vector<unsigned int>& body_part_pairs,
    const unsigned int number_body_parts,
    const unsigned int number_body_part_pairs,
    const double default_nms_threshold, const Array<double>& precomputed_pa_fs);

template void remove_people_below_thresholds_and_fill_faces(
    std::vector<int>& valid_subset_indexes, int& number_people,
    std::vector<std::pair<std::vector<int>, float>>& people_vector,
    const unsigned int number_body_parts, const int min_subset_cnt,
    const float min_subset_score, const bool maximize_positives,
    const float* const peaks_ptr);
template void remove_people_below_thresholds_and_fill_faces(
    std::vector<int>& valid_subset_indexes, int& number_people,
    std::vector<std::pair<std::vector<int>, double>>& people_vector,
    const unsigned int number_body_parts, const int min_subset_cnt,
    const double min_subset_score, const bool maximize_positives,
    const double* const peaks_ptr);

template void people_vector_to_people_array(
    Array<float>& pose_keypoints, Array<float>& pose_scores,
    const float scale_factor,
    const std::vector<std::pair<std::vector<int>, float>>& people_vector,
    const std::vector<int>& valid_subset_indexes, const float* const peaks_ptr,
    const int number_people, const unsigned int number_body_parts,
    const unsigned int number_body_part_pairs);
template void people_vector_to_people_array(
    Array<double>& pose_keypoints, Array<double>& pose_scores,
    const double scale_factor,
    const std::vector<std::pair<std::vector<int>, double>>& people_vector,
    const std::vector<int>& valid_subset_indexes, const double* const peaks_ptr,
    const int number_people, const unsigned int number_body_parts,
    const unsigned int number_body_part_pairs);

template std::vector<std::tuple<float, float, int, int, int>>
paf_ptr_into_vector(const Array<float>& pair_scores,
                    const float* const peaks_ptr, const int max_peaks,
                    const std::vector<unsigned int>& body_part_pairs,
                    const unsigned int number_body_part_pairs);
template std::vector<std::tuple<double, double, int, int, int>>
paf_ptr_into_vector(const Array<double>& pair_scores,
                    const double* const peaks_ptr, const int max_peaks,
                    const std::vector<unsigned int>& body_part_pairs,
                    const unsigned int number_body_part_pairs);

template std::vector<std::pair<std::vector<int>, float>>
paf_vector_into_people_vector(
    const std::vector<std::tuple<float, float, int, int, int>>&
        pair_connections,
    const float* const peaks_ptr, const int max_peaks,
    const std::vector<unsigned int>& body_part_pairs,
    const unsigned int number_body_parts);
template std::vector<std::pair<std::vector<int>, double>>
paf_vector_into_people_vector(
    const std::vector<std::tuple<double, double, int, int, int>>&
        pair_connections,
    const double* const peaks_ptr, const int max_peaks,
    const std::vector<unsigned int>& body_part_pairs,
    const unsigned int number_body_parts);

}  // namespace openposert
