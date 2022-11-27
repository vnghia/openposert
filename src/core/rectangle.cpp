#include "openposert/core/rectangle.hpp"

#include <stdexcept>

#include "openposert/core/common.hpp"
#include "openposert/core/macros.hpp"
#include "openposert/core/point.hpp"

namespace openposert {

template <typename T>
Rectangle<T>::Rectangle(const T x_, const T y_, const T width_, const T height_)
    : x{x_}, y{y_}, width{width_}, height{height_} {}

template <typename T>
Rectangle<T>::Rectangle(const Rectangle<T>& rectangle) {
  try {
    x = rectangle.x;
    y = rectangle.y;
    width = rectangle.width;
    height = rectangle.height;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
Rectangle<T>& Rectangle<T>::operator=(const Rectangle<T>& rectangle) {
  try {
    x = rectangle.x;
    y = rectangle.y;
    width = rectangle.width;
    height = rectangle.height;
    // return
    return *this;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return *this;
  }
}

template <typename T>
Rectangle<T>::Rectangle(Rectangle<T>&& rectangle) {
  try {
    x = rectangle.x;
    y = rectangle.y;
    width = rectangle.width;
    height = rectangle.height;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
Rectangle<T>& Rectangle<T>::operator=(Rectangle<T>&& rectangle) {
  try {
    x = rectangle.x;
    y = rectangle.y;
    width = rectangle.width;
    height = rectangle.height;
    // return
    return *this;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return *this;
  }
}

template <typename T>
Point<T> Rectangle<T>::center() const {
  try {
    return Point<T>{T(x + width / 2), T(y + height / 2)};
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return Point<T>{};
  }
}

template <typename T>
Point<T> Rectangle<T>::bottom_right() const {
  try {
    return Point<T>{T(x + width), T(y + height)};
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return Point<T>{};
  }
}

template <typename T>
void Rectangle<T>::recenter(const T new_width, const T new_height) {
  try {
    const auto center_point = center();
    x = center_point.x - T(new_width / 2.f);
    y = center_point.y - T(new_height / 2.f);
    width = new_width;
    height = new_height;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

template <typename T>
Rectangle<T>& Rectangle<T>::operator*=(const T value) {
  try {
    x *= value;
    y *= value;
    width *= value;
    height *= value;
    // return
    return *this;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return *this;
  }
}

template <typename T>
Rectangle<T> Rectangle<T>::operator*(const T value) const {
  try {
    return Rectangle<T>{T(x * value), T(y * value), T(width * value),
                        T(height * value)};
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return Rectangle<T>{};
  }
}

template <typename T>
Rectangle<T>& Rectangle<T>::operator/=(const T value) {
  try {
    x /= value;
    y /= value;
    width /= value;
    height /= value;
    // return
    return *this;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return *this;
  }
}

template <typename T>
Rectangle<T> Rectangle<T>::operator/(const T value) const {
  try {
    return Rectangle<T>{T(x / value), T(y / value), T(width / value),
                        T(height / value)};
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return Rectangle<T>{};
  }
}

COMPILE_TEMPLATE_BASIC_TYPES_STRUCT(Rectangle);

// static methods
template <typename T>
Rectangle<T> recenter(const Rectangle<T>& rectangle, const T new_width,
                      const T new_height) {
  try {
    Rectangle<T> result;
    const auto center_point = rectangle.center();
    result.x = center_point.x - T(new_width / 2.f);
    result.y = center_point.y - T(new_height / 2.f);
    result.width = new_width;
    result.height = new_height;
    return result;
  } catch (const std::exception& e) {
    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    return Rectangle<T>{};
  }
}

template Rectangle<char> recenter(const Rectangle<char>& rectangle,
                                  const char new_width, const char new_height);
template Rectangle<signed char> recenter(
    const Rectangle<signed char>& rectangle, const signed char new_width,
    const signed char new_height);
template Rectangle<short> recenter(const Rectangle<short>& rectangle,
                                   const short new_width,
                                   const short new_height);
template Rectangle<int> recenter(const Rectangle<int>& rectangle,
                                 const int new_width, const int new_height);
template Rectangle<long> recenter(const Rectangle<long>& rectangle,
                                  const long new_width, const long new_height);
template Rectangle<long long> recenter(const Rectangle<long long>& rectangle,
                                       const long long new_width,
                                       const long long new_height);
template Rectangle<unsigned char> recenter(
    const Rectangle<unsigned char>& rectangle, const unsigned char new_width,
    const unsigned char new_height);
template Rectangle<unsigned short> recenter(
    const Rectangle<unsigned short>& rectangle, const unsigned short new_width,
    const unsigned short new_height);
template Rectangle<unsigned int> recenter(
    const Rectangle<unsigned int>& rectangle, const unsigned int new_width,
    const unsigned int new_height);
template Rectangle<unsigned long> recenter(
    const Rectangle<unsigned long>& rectangle, const unsigned long new_width,
    const unsigned long new_height);
template Rectangle<unsigned long long> recenter(
    const Rectangle<unsigned long long>& rectangle,
    const unsigned long long new_width, const unsigned long long new_height);
template Rectangle<float> recenter(const Rectangle<float>& rectangle,
                                   const float new_width,
                                   const float new_height);
template Rectangle<double> recenter(const Rectangle<double>& rectangle,
                                    const double new_width,
                                    const double new_height);
template Rectangle<long double> recenter(
    const Rectangle<long double>& rectangle, const long double new_width,
    const long double new_height);

}  // namespace openposert
