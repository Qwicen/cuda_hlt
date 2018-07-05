#pragma once

#include "../../handlers/include/HandlerMaker.cuh"
#include <tuple>

template<typename T>
struct Sequence {
  T algorithms;

  Sequence() = default;
  
  Sequence(T param_algorithms) :
    algorithms(param_algorithms) {}

  template<typename U, unsigned long... Is>
  void set_helper(
    U u,
    std::index_sequence<Is...>
  ) {
    // Trick to assign to all elements
    auto l = {(std::get<Is>(algorithms) = HandlerMaker<Is>::make_handler(std::get<Is>(u)), 0)...};
  }

  /**
   * @brief Populates all elements in the sequence with the
   *        kernel functions passed.
   */
  template<typename... U>
  void set(U... u) {
    set_helper(std::make_tuple(u...), std::make_index_sequence<sizeof...(U)>());
  }

  template<unsigned long I>
  decltype(std::get<I>(algorithms)) item() noexcept {
    return std::get<I>(algorithms);
  }
};
