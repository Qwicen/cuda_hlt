#pragma once

#include <tuple>
#include "TupleTools.cuh"

/**
 * @brief A single argument.
 * @details Encapsulates the type, size and offset of the object
 * 
 * @tparam I [description]
 * @tparam T [description]
 */
template<int I, typename T>
struct Argument {
  constexpr static int i = I;
  T m_type_obj;
};

/**
 * @brief Helper class to generate arguments based on
 *        the information provided by the base_pointer and offsets
 */
template<typename T>
struct ArgumentManager {
  char* base_pointer;
  T arguments;
  std::array<size_t, std::tuple_size<T>::value> argument_sizes;
  std::array<uint, std::tuple_size<T>::value> argument_offsets;

  ArgumentManager(char* param_base_pointer)
  : base_pointer(param_base_pointer) {}

  template<unsigned I>
  auto offset() const
  -> decltype(std::get<I>(arguments).m_type_obj)* {
    auto& argument = std::get<I>(arguments);
    auto pointer = base_pointer + argument_offsets[I];
    return reinterpret_cast<decltype(argument.m_type_obj)*>(pointer);
  }

  template<unsigned I>
  size_t size() const {
    return argument_sizes[I];
  }

  template<unsigned I>
  void set_offset(uint offset) {
    argument_offsets[I] = offset;
  }

  template<unsigned I>
  void set_size(size_t size) {
    argument_sizes[I] = size * sizeof(std::get<I>(arguments).m_type_obj);
  }

  /**
   * @brief Support fetching the size by runtime argument
   *        instead of statically resolving it at compile time.
   *        This is needed for the current scheduler.
   */
  size_t size(const uint argument_index) const {
    return argument_sizes[argument_index];
  }

  /**
   * @brief Support setting the offset by runtime argument.
   *        This is needed for the current scheduler.
   */
  void set_offset(uint argument_index, uint offset) {
    argument_offsets[argument_index] = offset;
  }
};

/**
 * @brief Generates a std::vector with the sizes of all arguments,
 *        taking account of their types, in order
 */
template<typename T, unsigned long... Is>
std::vector<size_t> generate_argument_sizes_impl(
  const T& tuple,
  std::index_sequence<Is...>
) {
  return {std::get<Is>(tuple).size()...};
}

template<typename T>
std::vector<size_t> generate_argument_sizes(const T& tuple) {
  using indices = typename tuple_indices<T>::type;
  return generate_argument_sizes_impl(tuple, indices());
}

/**
 * @brief Generates a std::vector with the names of all arguments.
 */
template<typename T, unsigned long... Is>
std::vector<std::string> generate_argument_names_impl(
  const T& tuple,
  std::index_sequence<Is...>
) {
  return {std::get<Is>(tuple).name...};
}

template<typename T>
std::vector<std::string> generate_argument_names(const T& tuple) {
  using indices = typename tuple_indices<T>::type;
  return generate_argument_names_impl(tuple, indices());
}
