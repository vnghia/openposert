#include "openposert/core/array.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "openposert/core/common.hpp"
#include "openposert/core/macros.hpp"

namespace openposert {

template <typename T>
Array<T>::Array(const int size) {
  try {
    reset(size);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
Array<T>::Array(const std::vector<int>& sizes) {
  try {
    reset(sizes);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
Array<T>::Array(const int size, const T value) {
  try {
    reset(size, value);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
Array<T>::Array(const std::vector<int>& sizes, const T value) {
  try {
    reset(sizes, value);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
Array<T>::Array(const int size, T* const data_ptr) {
  try {
    reset(size, data_ptr);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
Array<T>::Array(const std::vector<int>& sizes, T* const data_ptr) {
  try {
    reset(sizes, data_ptr);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
Array<T>::Array(const Array<T>& array, const int index, const bool no_copy) {
  try {
    // sanity check
    if (array.get_size(0) <= index)
      error("index out of range.", __LINE__, __FUNCTION__, __FILE__);
    // define new size
    auto sizes = array.get_size();
    sizes[0] = 1;
    // move --> temporary Array<T> as long as `array` is in scope
    if (no_copy)
      reset_auxiliary(
          sizes, array.get_pseudo_const_ptr() + index * array.get_volume(1));
    // copy --> slower but it will always stay in scope
    else {
      // allocate memory
      reset(sizes);
      // copy desired index
      const auto array_area = (int)array.get_volume(1);
      const auto keypoints_index = index * array_area;
      std::copy(&array[keypoints_index], &array[keypoints_index] + array_area,
                data_.get());
    }
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
Array<T>::Array(const Array<T>& array)
    : size_(array.size_), volume_(array.volume_), data_(array.data_) {}

template <typename T>
Array<T>& Array<T>::operator=(const Array<T>& array) {
  try {
    size_ = array.size_;
    volume_ = array.volume_;
    data_ = array.data_;
    // return
    return *this;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return *this;
  }
}

template <typename T>
Array<T>::Array(Array<T>&& array) : size_(array.size_), volume_(array.volume_) {
  try {
    std::swap(data_, array.data_);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
Array<T>& Array<T>::operator=(Array<T>&& array) {
  try {
    size_ = array.size_;
    volume_ = array.volume_;
    std::swap(data_, array.data_);
    // return
    return *this;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return *this;
  }
}

template <typename T>
Array<T> Array<T>::clone() const {
  try {
    // constructor
    Array<T> array{size_};
    // clone data
    std::copy(data_.get(), data_.get() + volume_, array.data_.get());
    // return
    return array;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return Array<T>{};
  }
}

template <typename T>
void Array<T>::reset(const int size) {
  try {
    if (size > 0)
      reset(std::vector<int>{size});
    else
      reset(std::vector<int>{});
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
void Array<T>::reset(const std::vector<int>& sizes) {
  try {
    reset_auxiliary(sizes);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
void Array<T>::reset(const int sizes, const T value) {
  try {
    reset(sizes);
    set_to(value);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
void Array<T>::reset(const std::vector<int>& sizes, const T value) {
  try {
    reset(sizes);
    set_to(value);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
void Array<T>::reset(const int size, T* const data_ptr) {
  try {
    if (size > 0)
      reset_auxiliary(std::vector<int>{size}, data_ptr);
    else
      error("size cannot be less than 1.", __LINE__, __FUNCTION__, __FILE__);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
void Array<T>::reset(const std::vector<int>& sizes, T* const data_ptr) {
  try {
    if (!sizes.empty())
      reset_auxiliary(sizes, data_ptr);
    else
      error("size cannot be empty or less than 1.", __LINE__, __FUNCTION__,
            __FILE__);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
void Array<T>::set_to(const T value) {
  try {
    if (volume_ > 0) {
      for (auto i = 0u; i < volume_; i++) operator[](i) = value;
    }
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
int Array<T>::get_size(const int index) const {
  try {
    // matlab style:
    // if empty -> return 0
    // if index >= # dimensions -> return 1
    if ((unsigned int)index < size_.size() && 0 <= index)
      return size_[index];
    else
      return (!size_.empty());
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0;
  }
}

template <typename T>
size_t Array<T>::get_volume(const int index_a, const int index_b) const {
  try {
    const auto index_b_final =
        (index_b != -1 ? index_b : (int)size_.size() - 1);
    if (index_a < index_b_final) {
      // 0 <= index_a < index_b_final < size_.size()
      if (0 <= index_a && (unsigned int)index_b_final < size_.size())
        return std::accumulate(size_.begin() + index_a,
                               size_.begin() + index_b_final + 1, 1ull,
                               std::multiplies<size_t>());
      else {
        error("indexes out of dimension.", __LINE__, __FUNCTION__, __FILE__);
        return 0;
      }
    } else if (index_a == index_b_final)
      return size_.at(index_a);
    else  // if (index_a > index_b_final)
    {
      error("index_a > index_b.", __LINE__, __FUNCTION__, __FILE__);
      return 0;
    }
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0;
  }
}

template <typename T>
std::vector<int> Array<T>::get_stride() const {
  try {
    std::vector<int> strides(size_.size());
    if (!strides.empty()) {
      strides.back() = sizeof(T);
      for (auto i = (int)strides.size() - 2; i > -1; i--)
        strides[i] = strides[i + 1] * size_[i + 1];
    }
    return strides;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return {};
  }
}

template <typename T>
int Array<T>::get_stride(const int index) const {
  try {
    return get_stride()[index];
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return -1;
  }
}

template <typename T>
int Array<T>::get_index(const std::vector<int>& indexes) const {
  try {
    auto index = 0;
    auto accumulated = 1;
    for (auto i = (int)indexes.size() - 1; i >= 0; i--) {
      index += accumulated * indexes[i];
      accumulated *= size_[i];
    }
    return index;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0;
  }
}

template <typename T>
int Array<T>::get_index_and_check(const std::vector<int>& indexes) const {
  try {
    if (indexes.size() != size_.size())
      error("requested indexes size is different than array size.", __LINE__,
            __FUNCTION__, __FILE__);
    return get_index(indexes);
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return 0;
  }
}

template <typename T>
T& Array<T>::common_at(const int index) const {
  try {
    if (0 <= index && (size_t)index < volume_)
      return data_[index];  // data_.get()[index]
    else {
      error("index out of bounds: 0 <= index && index < volume_", __LINE__,
            __FUNCTION__, __FILE__);
      return data_[0];  // data_.get()[0]
    }
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return data_[0];  // data_.get()[0]
  }
}

template <typename T>
void Array<T>::reset_auxiliary(const std::vector<int>& sizes,
                               T* const data_ptr) {
  try {
    if (!sizes.empty()) {
      // new size & volume
      size_ = sizes;
      volume_ = {std::accumulate(sizes.begin(), sizes.end(), std::size_t(1),
                                 std::multiplies<size_t>())};
      // prepare shared_ptr
      if (data_ptr == nullptr) {
        data_.reset(new T[volume_]);
        // sanity check
        if (data_ == nullptr)
          error("shared pointer could not be allocated for array data storage.",
                __LINE__, __FUNCTION__, __FILE__);
      } else {
        // non-owning
        data_.reset(data_ptr, [](T* p) {});
      }
    } else {
      size_ = {};
      volume_ = 0ul;
      data_.reset();
    }
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

COMPILE_TEMPLATE_BASIC_TYPES_CLASS(Array);

}  // namespace openposert
