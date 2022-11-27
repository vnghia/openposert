#pragma once

namespace openposert {
// use op::round/max/min for basic types (int, char, long, float, double, etc).
// never with classes! `std::` alternatives uses 'const T&' instead of 'const T'
// as argument. e.g., std::round is really slow (~300 ms vs ~10 ms when i
// individually apply it to each element of a whole image array

// very important: these fast functions does not work for negative integer
// numbers. e.g., positive_int_round(-180.f) = -179.

// round functions
// signed

template <typename T>
inline char positive_char_round(const T a) {
  return char(a + 0.5f);
}

template <typename T>
inline signed char positive_s_char_round(const T a) {
  return (signed char)(a + 0.5f);
}

template <typename T>
inline int positive_int_round(const T a) {
  return int(a + 0.5f);
}

template <typename T>
inline long positive_long_round(const T a) {
  return long(a + 0.5f);
}

template <typename T>
inline long long positive_long_long_round(const T a) {
  return (long long)(a + 0.5f);
}

// unsigned
template <typename T>
inline unsigned char u_char_round(const T a) {
  return (unsigned char)(a + 0.5f);
}

template <typename T>
inline unsigned int u_int_round(const T a) {
  return (unsigned int)(a + 0.5f);
}

template <typename T>
inline unsigned long ulong_round(const T a) {
  return (unsigned long)(a + 0.5f);
}

template <typename T>
inline unsigned long long u_long_long_round(const T a) {
  return (unsigned long long)(a + 0.5f);
}

// max/min functions
template <typename T>
inline T fast_max(const T a, const T b) {
  return (a > b ? a : b);
}

template <typename T>
inline T fast_min(const T a, const T b) {
  return (a < b ? a : b);
}

template <class T>
inline T fast_truncate(T value, T min = 0, T max = 1) {
  return fast_min(max, fast_max(min, value));
}
}  // namespace openposert
