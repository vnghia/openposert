#include "openposert/utilities/keypoint.hpp"

#include <limits>

#include "openposert/core/common.hpp"
#include "openposert/core/point.hpp"
#include "openposert/utilities/fast_math.hpp"

namespace openposert {

const std::string error_message =
    "the Array<T> is not a rgb image or 3-channel keypoint array. this function"
    " is only for array of dimension: [size_a x size_b x 3].";

template <typename T>
T get_distance(const Array<T>& keypoints, const int person, const int element_a,
               const int element_b) {
  try {
    const auto keypoint_ptr =
        keypoints.get_const_ptr() +
        person * keypoints.get_size(1) * keypoints.get_size(2);
    const auto pixel_x =
        keypoint_ptr[element_a * 3] - keypoint_ptr[element_b * 3];
    const auto pixel_y =
        keypoint_ptr[element_a * 3 + 1] - keypoint_ptr[element_b * 3 + 1];
    return std::sqrt(pixel_x * pixel_x + pixel_y * pixel_y);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return T(-1);
  }
}

template float get_distance(const Array<float>& keypoints, const int person,
                            const int element_a, const int element_b);
template double get_distance(const Array<double>& keypoints, const int person,
                             const int element_a, const int element_b);

template <typename T>
void average_keypoints(Array<T>& keypoints_a, const Array<T>& keypoints_b,
                       const int person_a) {
  try {
    // sanity checks
    if (keypoints_a.get_number_dimensions() !=
        keypoints_b.get_number_dimensions())
      error(
          "keypoints_a.get_number_dimensions() != "
          "keypoints_b.get_number_dimensions().",
          __LINE__, __FUNCTION__, __FILE__);
    for (auto dimension = 1u; dimension < keypoints_a.get_number_dimensions();
         dimension++)
      if (keypoints_a.get_size(dimension) != keypoints_b.get_size(dimension))
        error("keypoints_a.get_size() != keypoints_b.get_size().", __LINE__,
              __FUNCTION__, __FILE__);
    // for each body part
    const auto number_parts = keypoints_a.get_size(1);
    for (auto part = 0; part < number_parts; part++) {
      const auto final_index_a =
          keypoints_a.get_size(2) * (person_a * number_parts + part);
      const auto final_index_b = keypoints_a.get_size(2) * part;
      if (keypoints_b[final_index_b + 2] - keypoints_a[final_index_a + 2] >
          T(0.05)) {
        keypoints_a[final_index_a] = keypoints_b[final_index_b];
        keypoints_a[final_index_a + 1] = keypoints_b[final_index_b + 1];
        keypoints_a[final_index_a + 2] = keypoints_b[final_index_b + 2];
      }
    }
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}
template void average_keypoints(Array<float>& keypoints_a,
                                const Array<float>& keypoints_b,
                                const int person_a);
template void average_keypoints(Array<double>& keypoints_a,
                                const Array<double>& keypoints_b,
                                const int person_a);

template <typename T>
void scale_keypoints(Array<T>& keypoints, const T scale) {
  try {
    if (!keypoints.empty() && scale != T(1)) {
      // error check
      if (keypoints.get_size(2) != 3 && keypoints.get_size(2) != 4)
        error(
            "the Array<T> is not a (x,y,score) or (x,y,z,score) format array. "
            "this"
            " function is only for those 2 dimensions: [size_a x size_b x "
            "3or4].",
            __LINE__, __FUNCTION__, __FILE__);
      // get #people and #parts
      const auto number_people = keypoints.get_size(0);
      const auto number_parts = keypoints.get_size(1);
      const auto xyz_channels = keypoints.get_size(2);
      // for each person
      for (auto person = 0; person < number_people; person++) {
        // for each body part
        for (auto part = 0; part < number_parts; part++) {
          const auto final_index =
              xyz_channels * (person * number_parts + part);
          for (auto xyz = 0; xyz < xyz_channels - 1; xyz++)
            keypoints[final_index + xyz] *= scale;
        }
      }
    }
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}
template void scale_keypoints(Array<float>& keypoints, const float scale);
template void scale_keypoints(Array<double>& keypoints, const double scale);

template <typename T>
void scale_keypoints2d(Array<T>& keypoints, const T scale_x, const T scale_y) {
  try {
    if (!keypoints.empty() && (scale_x != T(1) || scale_y != T(1))) {
      // error check
      if (keypoints.get_size(2) != 3)
        error(error_message, __LINE__, __FUNCTION__, __FILE__);
      // get #people and #parts
      const auto number_people = keypoints.get_size(0);
      const auto number_parts = keypoints.get_size(1);
      // for each person
      for (auto person = 0; person < number_people; person++) {
        // for each body part
        for (auto part = 0; part < number_parts; part++) {
          const auto final_index = 3 * (person * number_parts + part);
          keypoints[final_index] *= scale_x;
          keypoints[final_index + 1] *= scale_y;
        }
      }
    }
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}
template void scale_keypoints2d(Array<float>& keypoints, const float scale_x,
                                const float scale_y);
template void scale_keypoints2d(Array<double>& keypoints, const double scale_x,
                                const double scale_y);

template <typename T>
void scale_keypoints2d(Array<T>& keypoints, const T scale_x, const T scale_y,
                       const T offset_x, const T offset_y) {
  try {
    if (!keypoints.empty() && (scale_x != T(1) || scale_y != T(1) ||
                               offset_x != T(0) || offset_y != T(0))) {
      // error check
      if (keypoints.get_size(2) != 3)
        error(error_message, __LINE__, __FUNCTION__, __FILE__);
      // get #people and #parts
      const auto number_people = keypoints.get_size(0);
      const auto number_parts = keypoints.get_size(1);
      // for each person
      for (auto person = 0; person < number_people; person++) {
        // for each body part
        for (auto part = 0; part < number_parts; part++) {
          const auto final_index =
              keypoints.get_size(2) * (person * number_parts + part);
          keypoints[final_index] = keypoints[final_index] * scale_x + offset_x;
          keypoints[final_index + 1] =
              keypoints[final_index + 1] * scale_y + offset_y;
        }
      }
    }
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}
template void scale_keypoints2d(Array<float>& keypoints, const float scale_x,
                                const float scale_y, const float offset_x,
                                const float offset_y);
template void scale_keypoints2d(Array<double>& keypoints, const double scale_x,
                                const double scale_y, const double offset_x,
                                const double offset_y);

template <typename T>
Rectangle<T> get_keypoints_rectangle(const Array<T>& keypoints,
                                     const int person, const T threshold,
                                     const int first_index,
                                     const int last_index) {
  try {
    // params
    const auto number_keypoints = keypoints.get_size(1);
    const auto last_index_clean =
        (last_index < 0 ? number_keypoints : last_index);
    // sanity checks
    if (number_keypoints < 1)
      error("number body parts must be > 0.", __LINE__, __FUNCTION__, __FILE__);
    if (last_index_clean > number_keypoints)
      error(
          "the value of `last_index` must be less or equal than "
          "`number_keypoints`. currently: " +
              std::to_string(last_index_clean) + " vs. " +
              std::to_string(number_keypoints),
          __LINE__, __FUNCTION__, __FILE__);
    if (first_index > last_index_clean)
      error(
          "the value of `first_index` must be less or equal than `last_index`. "
          "currently: " +
              std::to_string(first_index) + " vs. " +
              std::to_string(last_index),
          __LINE__, __FUNCTION__, __FILE__);
    // define keypoint_ptr
    const auto keypoint_ptr =
        keypoints.get_const_ptr() +
        person * keypoints.get_size(1) * keypoints.get_size(2);
    T min_x = std::numeric_limits<T>::max();
    T max_x = std::numeric_limits<T>::lowest();
    T min_y = min_x;
    T max_y = max_x;
    for (auto part = first_index; part < last_index_clean; part++) {
      const auto score = keypoint_ptr[3 * part + 2];
      if (score > threshold) {
        const auto x = keypoint_ptr[3 * part];
        const auto y = keypoint_ptr[3 * part + 1];
        // set x
        if (max_x < x) max_x = x;
        if (min_x > x) min_x = x;
        // set y
        if (max_y < y) max_y = y;
        if (min_y > y) min_y = y;
      }
    }
    if (max_x >= min_x && max_y >= min_y)
      return Rectangle<T>{min_x, min_y, max_x - min_x, max_y - min_y};
    else
      return Rectangle<T>{};
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return Rectangle<T>{};
  }
}
template Rectangle<float> get_keypoints_rectangle(const Array<float>& keypoints,
                                                  const int person,
                                                  const float threshold,
                                                  const int first_index,
                                                  const int last_index);
template Rectangle<double> get_keypoints_rectangle(
    const Array<double>& keypoints, const int person, const double threshold,
    const int first_index, const int last_index);

template <typename T>
T get_average_score(const Array<T>& keypoints, const int person) {
  try {
    // sanity check
    if (person >= keypoints.get_size(0))
      error("person index out of bounds.", __LINE__, __FUNCTION__, __FILE__);
    // get average score
    T score = T(0);
    const auto number_keypoints = keypoints.get_size(1);
    const auto area = number_keypoints * keypoints.get_size(2);
    const auto person_offset = person * area;
    for (auto part = 0; part < number_keypoints; part++)
      score += keypoints[person_offset + part * keypoints.get_size(2) + 2];
    return score / number_keypoints;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return T(0);
  }
}
template float get_average_score(const Array<float>& keypoints,
                                 const int person);
template double get_average_score(const Array<double>& keypoints,
                                  const int person);

template <typename T>
T get_keypoints_area(const Array<T>& keypoints, const int person,
                     const T threshold) {
  try {
    return get_keypoints_rectangle(keypoints, person, threshold).area();
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return T(0);
  }
}
template float get_keypoints_area(const Array<float>& keypoints,
                                  const int person, const float threshold);
template double get_keypoints_area(const Array<double>& keypoints,
                                   const int person, const double threshold);

template <typename T>
int get_biggest_person(const Array<T>& keypoints, const T threshold) {
  try {
    if (!keypoints.empty()) {
      const auto number_people = keypoints.get_size(0);
      auto biggest_pose_index = -1;
      auto biggest_area = T(-1);
      for (auto person = 0; person < number_people; person++) {
        const auto new_person_area =
            get_keypoints_area(keypoints, person, threshold);
        if (new_person_area > biggest_area) {
          biggest_area = new_person_area;
          biggest_pose_index = person;
        }
      }
      return biggest_pose_index;
    } else
      return -1;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return -1;
  }
}
template int get_biggest_person(const Array<float>& keypoints,
                                const float threshold);
template int get_biggest_person(const Array<double>& keypoints,
                                const double threshold);

template <typename T>
int get_non_zero_keypoints(const Array<T>& keypoints, const int person,
                           const T threshold) {
  try {
    if (!keypoints.empty()) {
      // sanity check
      if (keypoints.get_size(0) <= person)
        error("person index out of range.", __LINE__, __FUNCTION__, __FILE__);
      // count keypoints
      auto non_zero_counter = 0;
      const auto base_index = person * (int)keypoints.get_volume(1, 2);
      for (auto part = 0; part < keypoints.get_size(1); part++)
        if (keypoints[base_index + 3 * part + 2] >= threshold)
          non_zero_counter++;
      return non_zero_counter;
    } else
      return 0;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0;
  }
}
template int get_non_zero_keypoints(const Array<float>& keypoints,
                                    const int person, const float threshold);
template int get_non_zero_keypoints(const Array<double>& keypoints,
                                    const int person, const double threshold);

template <typename T>
T get_distance_average(const Array<T>& keypoints, const int person_a,
                       const int person_b, const T threshold) {
  try {
    return get_distance_average(keypoints, person_a, keypoints, person_b,
                                threshold);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return T(0);
  }
}
template float get_distance_average(const Array<float>& keypoints,
                                    const int person_a, const int person_b,
                                    const float threshold);
template double get_distance_average(const Array<double>& keypoints,
                                     const int person_a, const int person_b,
                                     const double threshold);

template <typename T>
T get_distance_average(const Array<T>& keypoints_a, const int person_a,
                       const Array<T>& keypoints_b, const int person_b,
                       const T threshold) {
  try {
    // sanity checks
    if (keypoints_a.get_size(0) <= person_a)
      error("person_a index out of range.", __LINE__, __FUNCTION__, __FILE__);
    if (keypoints_b.get_size(0) <= person_b)
      error("person_b index out of range.", __LINE__, __FUNCTION__, __FILE__);
    if (keypoints_a.get_size(1) != keypoints_b.get_size(1))
      error("keypoints should have the same number of keypoints.", __LINE__,
            __FUNCTION__, __FILE__);
    // get total distance
    T total_distance = 0;
    int non_zero_counter = 0;
    const auto base_index_a = person_a * (int)keypoints_a.get_volume(1, 2);
    const auto base_index_b = person_b * (int)keypoints_b.get_volume(1, 2);
    for (auto part = 0; part < keypoints_a.get_size(1); part++) {
      if (keypoints_a[base_index_a + 3 * part + 2] >= threshold &&
          keypoints_b[base_index_b + 3 * part + 2] >= threshold) {
        const auto x = keypoints_a[base_index_a + 3 * part] -
                       keypoints_b[base_index_b + 3 * part];
        const auto y = keypoints_a[base_index_a + 3 * part + 1] -
                       keypoints_b[base_index_b + 3 * part + 1];
        total_distance += T(std::sqrt(x * x + y * y));
        non_zero_counter++;
      }
    }
    // get distance average
    return total_distance / non_zero_counter;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return T(0);
  }
}
template float get_distance_average(const Array<float>& keypoints_a,
                                    const int person_a,
                                    const Array<float>& keypoints_b,
                                    const int person_b, const float threshold);
template double get_distance_average(const Array<double>& keypoints_a,
                                     const int person_a,
                                     const Array<double>& keypoints_b,
                                     const int person_b,
                                     const double threshold);

template <typename T>
Array<T> get_keypoints_person(const Array<T>& keypoints, const int person,
                              const bool no_copy) {
  try {
    return Array<T>(keypoints, person, no_copy);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return Array<T>{};
  }
}
template Array<float> get_keypoints_person(const Array<float>& keypoints,
                                           const int person,
                                           const bool no_copy);
template Array<double> get_keypoints_person(const Array<double>& keypoints,
                                            const int person,
                                            const bool no_copy);

template <typename T>
float get_keypoints_roi(const Array<T>& keypoints, const int person_a,
                        const int person_b, const T threshold) {
  try {
    return get_keypoints_roi(keypoints, person_a, keypoints, person_b,
                             threshold);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0.f;
  }
}
template float get_keypoints_roi(const Array<float>& keypoints,
                                 const int person_a, const int person_b,
                                 const float threshold);
template float get_keypoints_roi(const Array<double>& keypoints,
                                 const int person_a, const int person_b,
                                 const double threshold);

template <typename T>
float get_keypoints_roi(const Array<T>& keypoints_a, const int person_a,
                        const Array<T>& keypoints_b, const int person_b,
                        const T threshold) {
  try {
    // sanity checks
    if (keypoints_a.get_size(0) <= person_a)
      error("person_a index out of range.", __LINE__, __FUNCTION__, __FILE__);
    if (keypoints_b.get_size(0) <= person_b)
      error("person_b index out of range.", __LINE__, __FUNCTION__, __FILE__);
    if (keypoints_a.get_size(1) != keypoints_b.get_size(1))
      error("keypoints should have the same number of keypoints.", __LINE__,
            __FUNCTION__, __FILE__);
    // get roi
    const auto rectangle_a =
        get_keypoints_rectangle(keypoints_a, person_a, threshold);
    const auto rectangle_b =
        get_keypoints_rectangle(keypoints_b, person_b, threshold);
    return get_keypoints_roi(rectangle_a, rectangle_b);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0.f;
  }
}
template float get_keypoints_roi(const Array<float>& keypoints_a,
                                 const int person_a,
                                 const Array<float>& keypoints_b,
                                 const int person_b, const float threshold);
template float get_keypoints_roi(const Array<double>& keypoints_a,
                                 const int person_a,
                                 const Array<double>& keypoints_b,
                                 const int person_b, const double threshold);

template <typename T>
float get_keypoints_roi(const Rectangle<T>& rectangle_a,
                        const Rectangle<T>& rectangle_b) {
  try {
    // check if negative values, then normalize it
    auto rectangle_a_norm = rectangle_a;
    auto rectangle_b_norm = rectangle_b;
    // e.g., [-10,-10,w1,h1] and [-20,-20,w2,h2] should be equivalent to
    // [10,10,w1,h1] and [0,0,w2,h2]
    const auto bias_x = std::min(std::min(T{0}, rectangle_a.x), rectangle_b.x);
    if (bias_x != 0) {
      rectangle_a_norm.x -= bias_x;
      rectangle_b_norm.x -= bias_x;
    }
    const auto bias_y = std::min(std::min(T{0}, rectangle_a.y), rectangle_b.y);
    if (bias_y != 0) {
      rectangle_a_norm.y -= bias_y;
      rectangle_b_norm.y -= bias_y;
    }
    // get roi
    const Point<T> point_a_intersection{
        fast_max(rectangle_a_norm.x, rectangle_b_norm.x),
        fast_max(rectangle_a_norm.y, rectangle_b_norm.y)};
    const Point<T> point_b_intersection{
        fast_min(rectangle_a_norm.x + rectangle_a_norm.width,
                 rectangle_b_norm.x + rectangle_b_norm.width),
        fast_min(rectangle_a_norm.y + rectangle_a_norm.height,
                 rectangle_b_norm.y + rectangle_b_norm.height)};
    // make sure there is overlap
    if (point_a_intersection.x < point_b_intersection.x &&
        point_a_intersection.y < point_b_intersection.y) {
      const Rectangle<T> rectangle_intersection{
          point_a_intersection.x, point_a_intersection.y,
          point_b_intersection.x - point_a_intersection.x,
          point_b_intersection.y - point_a_intersection.y};
      const auto area_a = rectangle_a_norm.area();
      const auto area_b = rectangle_b_norm.area();
      const auto intersection = rectangle_intersection.area();
      return float(intersection) / float(area_a + area_b - intersection);
    }
    // if non overlap --> return 0
    else
      return 0.f;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0.f;
  }
}

template float get_keypoints_roi(const Rectangle<int>& rectangle_a,
                                 const Rectangle<int>& rectangle_b);
template float get_keypoints_roi(const Rectangle<unsigned int>& rectangle_a,
                                 const Rectangle<unsigned int>& rectangle_b);
template float get_keypoints_roi(const Rectangle<float>& rectangle_a,
                                 const Rectangle<float>& rectangle_b);
template float get_keypoints_roi(const Rectangle<double>& rectangle_a,
                                 const Rectangle<double>& rectangle_b);

}  // namespace openposert
