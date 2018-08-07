#pragma once

#include "HandlerMaker.cuh"
#include <tuple>

template<typename T>
struct Sequence {
  T algorithms;

  Sequence() = default;
  
  Sequence(T param_algorithms) :
    algorithms(param_algorithms) {}

  /**
   * @brief Populates the tuple of algorithms.
   */
  template<typename U>
  void set(U u) {
    algorithms = u;
  }

  template<unsigned long I>
  decltype(std::get<I>(algorithms)) item() noexcept {
    return std::get<I>(algorithms);
  }
};

template<typename U, unsigned long... Is>
constexpr auto make_algorithm_tuple_helper(
  U u,
  std::index_sequence<Is...>
) {
  // Trick to assign to all elements
  return std::make_tuple(HandlerMaker<Is>::make_handler(std::get<Is>(u))...);
}

/**
 * @brief Populates all elements in the sequence with the
 *        kernel functions passed.
 */
template<typename U>
constexpr auto make_algorithm_tuple(U u) {
  return make_algorithm_tuple_helper(
    u,
    std::make_index_sequence<std::tuple_size<U>::value>()
  );
}
