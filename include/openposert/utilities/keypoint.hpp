#pragma once

#include <vector>

#include "openposert/core/array.hpp"
#include "openposert/core/common.hpp"
#include "openposert/core/rectangle.hpp"

namespace openposert {

template <typename T>
T get_distance(const Array<T>& keypoints, const int person, const int element_a,
               const int element_b);

template <typename T>
void average_keypoints(Array<T>& keypoints_a, const Array<T>& keypoints_b,
                       const int person_a);

template <typename T>
void scale_keypoints(Array<T>& keypoints, const T scale);

template <typename T>
void scale_keypoints2d(Array<T>& keypoints, const T scale_x, const T scale_y);

template <typename T>
void scale_keypoints2d(Array<T>& keypoints, const T scale_x, const T scale_y,
                       const T offset_x, const T offset_y);

template <typename T>
Rectangle<T> get_keypoints_rectangle(const Array<T>& keypoints,
                                     const int person, const T threshold,
                                     const int first_index = 0,
                                     const int last_index = -1);

template <typename T>
T get_average_score(const Array<T>& keypoints, const int person);

template <typename T>
T get_keypoints_area(const Array<T>& keypoints, const int person,
                     const T threshold);

template <typename T>
int get_biggest_person(const Array<T>& keypoints, const T threshold);

template <typename T>
int get_non_zero_keypoints(const Array<T>& keypoints, const int person,
                           const T threshold);

template <typename T>
T get_distance_average(const Array<T>& keypoints, const int person_a,
                       const int person_b, const T threshold);

template <typename T>
T get_distance_average(const Array<T>& keypoints_a, const int person_a,
                       const Array<T>& keypoints_b, const int person_b,
                       const T threshold);

/**
 * creates and Array<T> with a specific person.
 * @param keypoints Array<T> with the original data array to slice.
 * @param person indicates the index of the array to extract.
 * @param no_copy indicates whether to perform a copy. copy will never go to
 * undefined behavior, however, if no_copy == true, then:
 *     1. it is faster, as no data copy is involved, but...
 *     2. if the array keypoints goes out of scope, then the resulting array
 * will provoke an undefined behavior.
 *     3. if the returned array is modified, the information in the array
 * keypoints will also be.
 * @return Array<T> with the same dimension than keypoints expect the first
 * dimension being 1. e.g., if keypoints is {p,k,m}, the resulting Array<T> is
 * {1,k,m}.
 */
template <typename T>
Array<T> get_keypoints_person(const Array<T>& keypoints, const int person,
                              const bool no_copy = false);

template <typename T>
float get_keypoints_roi(const Array<T>& keypoints, const int person_a,
                        const int person_b, const T threshold);

template <typename T>
float get_keypoints_roi(const Array<T>& keypoints_a, const int person_a,
                        const Array<T>& keypoints_b, const int person_b,
                        const T threshold);

template <typename T>
float get_keypoints_roi(const Rectangle<T>& rectangle_a,
                        const Rectangle<T>& rectangle_b);

}  // namespace openposert
