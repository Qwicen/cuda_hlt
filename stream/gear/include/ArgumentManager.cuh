#pragma once

#include <tuple>
#include "TupleTools.cuh"

/**
 * @brief Helper class to generate arguments based on
 *        the information provided by the base_pointer and offsets
 */
template<typename Tuple>
struct ArgumentManager {
  char* base_pointer;
  std::array<size_t, std::tuple_size<Tuple>::value> argument_sizes;
  std::array<uint, std::tuple_size<Tuple>::value> argument_offsets;

  ArgumentManager() = default;

  void set_base_pointer(char* param_base_pointer) {
    base_pointer = param_base_pointer;
  }

  template<typename T>
  auto offset() const {
    auto pointer = base_pointer + argument_offsets[tuple_contains<T, Tuple>::index];
    return reinterpret_cast<typename T::type*>(pointer);
  }

  template<typename T>
  size_t size() const {
    return argument_sizes[tuple_contains<T, Tuple>::index];
  }

  template<typename T>
  void set_offset(uint offset) {
    argument_offsets[tuple_contains<T, Tuple>::index] = offset;
  }

  template<typename T>
  void set_size(size_t size) {
    argument_sizes[tuple_contains<T, Tuple>::index] = size * sizeof(typename T::type);
  }
};
