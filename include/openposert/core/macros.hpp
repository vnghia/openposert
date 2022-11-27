#pragma once

#define COMPILE_TEMPLATE_BASIC_TYPES_CLASS(class_name) \
  COMPILE_TEMPLATE_BASIC_TYPES(class_name, class)
#define COMPILE_TEMPLATE_BASIC_TYPES_STRUCT(class_name) \
  COMPILE_TEMPLATE_BASIC_TYPES(class_name, struct)
#define COMPILE_TEMPLATE_BASIC_TYPES(class_name, class_type) \
  template class_type class_name<char>;                      \
  template class_type class_name<signed char>;               \
  template class_type class_name<short>;                     \
  template class_type class_name<int>;                       \
  template class_type class_name<long>;                      \
  template class_type class_name<long long>;                 \
  template class_type class_name<unsigned char>;             \
  template class_type class_name<unsigned short>;            \
  template class_type class_name<unsigned int>;              \
  template class_type class_name<unsigned long>;             \
  template class_type class_name<unsigned long long>;        \
  template class_type class_name<float>;                     \
  template class_type class_name<double>;                    \
  template class_type class_name<long double>
