#pragma once

#include "Handler.cuh"
#include <tuple>

template<typename... T>
struct Sequence {
  std::tuple<T...> algorithms;

  Sequence() = default;
  Sequence(T... param_algorithms) :
    algorithms(std::tuple<T...>{param_algorithms...}) {}

  // TODO: For now, let's stick to unsigned i access
  // template<typename Fn>
  // auto& item(Fn fn) noexcept {
  //   return std::get<decltype(generate_handler(fn))>(algorithms);
  // }

  template<unsigned long i>
  decltype(std::get<i>(algorithms)) item() noexcept {
    return std::get<i>(algorithms);
  }
};

template<typename... T>
Sequence<T...> generate_sequence(T... t) {
  return Sequence<T...>{t...};
}

template<typename T>
struct Argument {
  T type_obj;
  std::string name;
  size_t size;

  // Argument() = default;
  Argument(const std::string& param_name, const size_t param_size)
    : name(param_name), size(param_size) {}
};

/**
 * @brief Helper to generate a tuple without specifying its type
 */
template<typename... T>
std::tuple<T...> generate_tuple(T... t) {
  return {t...};
}

/**
 * @brief Helper class to generate arguments based on
 *        the information provided by the base_pointer and offsets
 */
template<typename T>
struct ArgumentGenerator {
  T& arguments;
  char* base_pointer;
  std::vector<uint> offsets;

  ArgumentGenerator(T& param_arguments,
    char* param_base_pointer,
    std::vector<uint> param_offsets)
  : arguments(param_arguments),
    base_pointer(param_base_pointer),
    offsets(param_offsets) {}

  template<unsigned I>
  auto generate()
  -> decltype(std::get<I>(arguments).type_obj)* {
    auto& argument = std::get<I>(arguments);
    auto pointer = base_pointer + offsets[I];
    return reinterpret_cast<decltype(argument.type_obj)*>(pointer);
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
  return {std::get<Is>(tuple).size * sizeof(std::get<Is>(tuple).type_obj)...};
}

template<typename T>
std::vector<size_t> generate_argument_sizes(const T& tuple) {
  using indices = typename tuple_indices<T>::type;
  return generate_argument_sizes_impl(tuple, indices());
}
