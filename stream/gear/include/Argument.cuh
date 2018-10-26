#pragma once

#include <tuple>
#include "TupleTools.cuh"

/**
 * @brief Macro for defining arguments. An argument has an identifier
 *        and a type.
 */
#define ARGUMENT(ARGUMENT_NAME, ARGUMENT_TYPE) \
  struct ARGUMENT_NAME {\
    constexpr static auto name {#ARGUMENT_NAME};\
    using type = ARGUMENT_TYPE;\
  };

/**
 * @brief Defines dependencies for an algorithm.
 * 
 * @tparam T The algorithm type.
 * @tparam Args The dependencies.
 */
template<typename T, typename... Args>
struct AlgorithmDependencies {
  using Algorithm = T;
  using Arguments = std::tuple<Args...>;
};

/**
 * @brief Helper class to generate arguments based on
 *        the information provided by the base_pointer and offsets
 */
template<typename Tuple>
struct ArgumentManager {
  char* base_pointer;
  Tuple arguments;
  std::array<size_t, std::tuple_size<Tuple>::value> argument_sizes;
  std::array<uint, std::tuple_size<Tuple>::value> argument_offsets;

  ArgumentManager(char* param_base_pointer)
  : base_pointer(param_base_pointer) {}

  template<typename T>
  auto offset() const
  -> T::type* {
    auto pointer = base_pointer + argument_offsets[tuple_contains<T, Tuple>::index];
    return reinterpret_cast<T::type*>(pointer);
  }

  template<typename T>
  size_t size() const {
    return argument_sizes[tuple_contains<T, Tuple>::index];
  }

  template<typename T>
  void set_offset(uint offset) {
    argument_offsets[tuple_containts<T, Tuple>::index] = offset;
  }

  template<typename T>
  void set_size(size_t size) {
    argument_sizes[tuple_contains<T, Tuple>::index] = size * sizeof(T::type);
  }
};
