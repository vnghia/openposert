#pragma once

#include <memory>
#include <vector>

#include "openposert/core/common.hpp"

namespace openposert {
/**
 * array<T>: the open_pose basic raw data container
 * this template class implements a multidimensional data array. it is our basic
 * data container, analogous to mat in open_cv, tensor in torch/tensor_flow or
 * blob in caffe. it wraps a matrix and a std::shared_ptr, both of them pointing
 * to the same raw data. i.e. they both share the same memory, so we can read
 * and modify this data in both formats with no performance impact. hence, it
 * keeps high performance while adding high-level functions.
 */
template <typename T>
class Array {
 public:
  // ------------------------------ constructors and data allocator functions
  // ------------------------------ //
  /**
   * Array constructor.
   * equivalent to default constructor + reset(const int size).
   * @param size integer with the number of t element to be allocated. e.g.,
   * size = 5 is internally similar to `new t[5]`.
   */
  explicit Array(const int size);

  /**
   * Array constructor.
   * equivalent to default constructor + reset(const std::vector<int>& size =
   * {}).
   * @param sizes vector with the size of each dimension. e.g., size = {3, 5, 2}
   * is internally similar to `new t[3*5*2]`.
   */
  explicit Array(const std::vector<int>& sizes = {});

  /**
   * Array constructor.
   * equivalent to default constructor + reset(const int size, const T value).
   * @param size integer with the number of t element to be allocated. e.g.,
   * size = 5 is internally similar to `new t[5]`.
   * @param value initial value for each component of the array.
   */
  Array(const int size, const T value);

  /**
   * Array constructor.
   * equivalent to default constructor + reset(const std::vector<int>& size,
   * const T value).
   * @param sizes vector with the size of each dimension. e.g., size = {3, 5, 2}
   * is internally similar to: `new t[3*5*2]`.
   * @param value initial value for each component of the array.
   */
  Array(const std::vector<int>& sizes, const T value);

  /**
   * Array constructor.
   * equivalent to default constructor, but it does not allocate memory, but
   * rather use data_ptr.
   * @param size integer with the number of t element to be allocated. e.g.,
   * size = 5 is internally similar to `new t[5]`.
   * @param data_ptr pointer to the memory to be used by the array.
   */
  Array(const int size, T* const data_ptr);

  /**
   * Array constructor.
   * equivalent to default constructor, but it does not allocate memory, but
   * rather use data_ptr.
   * @param sizes vector with the size of each dimension. e.g., size = {3, 5, 2}
   * is internally similar to: `new t[3*5*2]`.
   * @param data_ptr pointer to the memory to be used by the array.
   */
  Array(const std::vector<int>& sizes, T* const data_ptr);

  /**
   * array constructor.
   * @param array array<T> with the original data array to slice.
   * @param index indicates the index of the array to extract.
   * @param no_copy indicates whether to perform a copy. copy will never go to
   * undefined behavior, however, if no_copy == true, then:
   *     1. it is faster, as no data copy is involved, but...
   *     2. if the array array goes out of scope, then the resulting array will
   * provoke an undefined behavior.
   *     3. if the returned array is modified, the information in the array
   * array will also be.
   * @return array<T> with the same dimension than array expect the first
   * dimension being 1. e.g., if array is {p,k,m}, the resulting array<T> is
   * {1,k,m}.
   */
  Array(const Array<T>& array, const int index, const bool no_copy = false);

  /**
   * array constructor. it manually copies the array<T2> into the new array<T>
   * @param array array<T2> with a format t2 different to the current array type
   * t.
   */
  template <typename T2>
  Array(const Array<T2>& array) : Array{array.get_size()} {
    try {
      // copy
      for (auto i = 0u; i < array.get_volume(); i++) data_[i] = t(array[i]);
    } catch (const std::exception& e) {
      error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
  }

  /**
   * copy constructor.
   * it performs `fast copy`: for performance purpose, copying a array<T> or
   * datum or cv::mat just copies the reference, it still shares the same
   * internal data. modifying the copied element will modify the original one.
   * use clone() for a slower but real copy, similarly to cv::mat and array<T>.
   * @param array array to be copied.
   */
  Array(const Array<T>& array);

  /**
   * copy assignment.
   * similar to array<T>(const array<T>& array).
   * @param array array to be copied.
   * @return the resulting array.
   */
  Array<T>& operator=(const Array<T>& array);

  /**
   * move constructor.
   * it destroys the original array to be moved.
   * @param array array to be moved.
   */
  Array(Array<T>&& array);

  /**
   * move assignment.
   * similar to array<T>(array<T>&& array).
   * @param array array to be moved.
   * @return the resulting array.
   */
  Array<T>& operator=(Array<T>&& array);

  /**
   * clone function.
   * similar to cv::mat::clone and datum::clone.
   * it performs a real but slow copy of the data, i.e., even if the copied
   * element is modified, the original one is not.
   * @return the resulting array.
   */
  Array<T> clone() const;

  /**
   * data allocation function.
   * it allocates the required space for the memory (it does not initialize that
   * memory).
   * @param size integer with the number of t element to be allocated. e.g.,
   * size = 5 is internally similar to `new t[5]`.
   */
  void reset(const int size);

  /**
   * data allocation function.
   * similar to reset(const int size), but it allocates a multi-dimensional
   * array of dimensions each of the values of the argument.
   * @param sizes vector with the size of each dimension. e.g., size = {3, 5, 2}
   * is internally similar to `new t[3*5*2]`.
   */
  void reset(const std::vector<int>& sizes = {});

  /**
   * data allocation function.
   * similar to reset(const int size), but initializing the data to the value
   * specified by the second argument.
   * @param size integer with the number of t element to be allocated. e.g.,
   * size = 5 is internally similar to `new t[5]`.
   * @param value initial value for each component of the array.
   */
  void reset(const int size, const T value);

  /**
   * data allocation function.
   * similar to reset(const std::vector<int>& size), but initializing the data
   * to the value specified by the second argument.
   * @param sizes vector with the size of each dimension. e.g., size = {3, 5, 2}
   * is internally similar to `new t[3*5*2]`.
   * @param value initial value for each component of the array.
   */
  void reset(const std::vector<int>& sizes, const T value);

  /**
   * data allocation function.
   * equivalent to default constructor, but it does not allocate memory, but
   * rather use data_ptr.
   * @param size integer with the number of t element to be allocated. e.g.,
   * size = 5 is internally similar to `new t[5]`.
   * @param data_ptr pointer to the memory to be used by the array.
   */
  void reset(const int size, T* const data_ptr);

  /**
   * data allocation function.
   * equivalent to default constructor, but it does not allocate memory, but
   * rather use data_ptr.
   * @param sizes vector with the size of each dimension. e.g., size = {3, 5, 2}
   * is internally similar to: `new t[3*5*2]`.
   * @param data_ptr pointer to the memory to be used by the array.
   */
  void reset(const std::vector<int>& sizes, T* const data_ptr);

  /**
   * data allocation function.
   * it internally assigns all the allocated memory to the value indicated by
   * the argument.
   * @param value value for each component of the array.
   */
  void set_to(const T value);

  // ------------------------------ data information functions
  // ------------------------------ //
  /**
   * check whether memory has been allocated.
   * @return true if no memory has been allocated, false otherwise.
   */
  inline bool empty() const { return (volume_ == 0); }

  /**
   * return a vector with the size of each dimension allocated.
   * @return a std::vector<int> with the size of each dimension. if no memory
   * has been allocated, it will return an empty std::vector.
   */
  inline std::vector<int> get_size() const { return size_; }

  /**
   * return a vector with the size of the desired dimension.
   * @param index dimension to check its size.
   * @return size of the desired dimension. it will return 0 if the requested
   * dimension is higher than the number of dimensions.
   */
  int get_size(const int index) const;

  /**
   * return the total number of dimensions, equivalent to get_size().size().
   * @return the number of dimensions. if no memory is allocated, it returns 0.
   */
  inline size_t get_number_dimensions() const { return size_.size(); }

  /**
   * return the total number of elements allocated, equivalent to multiply all
   * the components from get_size(). e.g., for a array<T> of size = {2,5,3}, the
   * volume or total number of elements is: 2x5x3 = 30.
   * @return the total volume of the allocated data. if no memory is allocated,
   * it returns 0.
   */
  inline size_t get_volume() const { return volume_; }

  /**
   * similar to get_volume(), but in this case it just returns the volume
   * between the desired dimensions. e.g., for a array<T> of size = {2,5,3}, the
   * volume or total number of elements for get_volume(1,2) is 5x3 = 15.
   * @param index_a dimension where to start.
   * @param index_b dimension where to stop. if index_b == -1, then it will take
   * up to the last dimension.
   * @return the total volume of the allocated data between the desired
   * dimensions. if the index are out of bounds, it throws an error.
   */
  size_t get_volume(const int index_a, const int index_b = -1) const;

  /**
   * return the stride or step size of the array.
   * e.g., given and array<T> of size 5x3, get_stride() would return the
   * following vector: {5x3sizeof(t), 3sizeof(t), sizeof(t)}.
   */
  std::vector<int> get_stride() const;

  /**
   * return the stride or step size of the array at the index-th dimension.
   * e.g., given and array<T> of size 5x3, get_stride(2) would return sizeof(t).
   */
  int get_stride(const int index) const;

  // ------------------------------ data access functions and operators
  // ------------------------------ //
  /**
   * return a raw pointer to the data. similar to: std::shared_ptr::get().
   * note: if you modify the pointer data, you will directly modify it in the
   * array<T> instance too. if you know you do not want to modify the data, then
   * use get_const_ptr() instead.
   * @return a raw pointer to the data.
   */
  inline T* get_ptr() { return data_.get(); }

  /**
   * similar to get_ptr(), but it forbids the data to be edited.
   * @return a raw const pointer to the data.
   */
  inline const T* get_const_ptr() const { return data_.get(); }

  /**
   * similar to get_const_ptr(), but it allows the data to be edited.
   * this function is only implemented for pybind11 usage.
   * @return a raw pointer to the data.
   */
  inline T* get_pseudo_const_ptr() const { return data_.get(); }

  /**
   * [] operator
   * similar to the [] operator for raw pointer data.
   * if debug mode is enabled, then it will check that the desired index is in
   * the data range, and it will throw an exception otherwise (similar to the at
   * operator).
   * @param index the desired memory location.
   * @return a editable reference to the data on the desired index location.
   */
  inline T& operator[](const int index) {
#ifdef ndebug
    return data_[index];
#else
    return at(index);
#endif
  }

  /**
   * [] operator
   * same functionality as operator[](const int index), but it forbids modifying
   * the value. otherwise, const functions would not be able to call the []
   * operator.
   * @param index the desired memory location.
   * @return a non-editable reference to the data on the desired index location.
   */
  inline const T& operator[](const int index) const {
#ifdef ndebug
    return data_[index];
    data_[index]
#else
    return at(index);
#endif
  }

  /**
   * [] operator
   * same functionality as operator[](const int index), but it lets the user
   * introduce the multi-dimensional index. e.g., given a (10 x 10 x 10) array,
   * array[11] is equivalent to array[{1,1,0}]
   * @param indexes vector with the desired memory location.
   * @return a editable reference to the data on the desired index location.
   */
  inline T& operator[](const std::vector<int>& indexes) {
    return operator[](get_index(indexes));
  }

  /**
   * [] operator
   * same functionality as operator[](const std::vector<int>& indexes), but it
   * forbids modifying the value. otherwise, const functions would not be able
   * to call the [] operator.
   * @param indexes vector with the desired memory location.
   * @return a non-editable reference to the data on the desired index location.
   */
  inline const T& operator[](const std::vector<int>& indexes) const {
    return operator[](get_index(indexes));
  }

  /**
   * at() function
   * same functionality as operator[](const int index), but it always check
   * whether the indexes are within the data bounds. otherwise, it will throw an
   * error.
   * @param index the desired memory location.
   * @return a editable reference to the data on the desired index location.
   */
  inline T& at(const int index) { return common_at(index); }

  /**
   * at() function
   * same functionality as operator[](const int index) const, but it always
   * check whether the indexes are within the data bounds. otherwise, it will
   * throw an error.
   * @param index the desired memory location.
   * @return a non-editable reference to the data on the desired index location.
   */
  inline const T& at(const int index) const { return common_at(index); }

  /**
   * at() function
   * same functionality as operator[](const std::vector<int>& indexes), but it
   * always check whether the indexes are within the data bounds. otherwise, it
   * will throw an error.
   * @param indexes vector with the desired memory location.
   * @return a editable reference to the data on the desired index location.
   */
  inline T& at(const std::vector<int>& indexes) {
    return at(get_index_and_check(indexes));
  }

  /**
   * at() function
   * same functionality as operator[](const std::vector<int>& indexes) const,
   * but it always check whether the indexes are within the data bounds.
   * otherwise, it will throw an error.
   * @param indexes vector with the desired memory location.
   * @return a non-editable reference to the data on the desired index location.
   */
  inline const T& at(const std::vector<int>& indexes) const {
    return at(get_index_and_check(indexes));
  }

 private:
  std::vector<int> size_;
  size_t volume_;
  std::shared_ptr<T[]> data_;

  /**
   * auxiliary function that both operator[](const std::vector<int>& indexes)
   * and operator[](const std::vector<int>& indexes) const use. it turn the
   * multi-dimensions indexes into the 1-dimension equivalent index.
   * @param indexes vector with the desired memory location.
   * @return the equivalent 1-d index.
   */
  int get_index(const std::vector<int>& indexes) const;

  /**
   * similar to get_index(const std::vector<int>& indexes) const, but used for
   * at(const std::vector<int>& indexes) and at(const std::vector<int>& indexes)
   * const. it also checks whether the index is within the allocated memory.
   * @param indexes vector with the desired memory location.
   * @return the equivalent 1-d index.
   */
  int get_index_and_check(const std::vector<int>& indexes) const;

  /**
   * auxiliary function that both at(const int index) and at(const int index)
   * const use.
   * @param index the desired memory location.
   * @return a non-editable reference to the data on the desired index location.
   */
  T& common_at(const int index) const;

  void reset_auxiliary(const std::vector<int>& sizes,
                       T* const data_ptr = nullptr);
};

}  // namespace openposert
