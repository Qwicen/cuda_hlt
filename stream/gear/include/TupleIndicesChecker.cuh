#pragma once

#include "TupleTools.cuh"

template<typename T>
struct TupleIndicesChecker {
  template<unsigned long... Is>
  constexpr static bool check_tuple_indices_impl(
    std::index_sequence<Is...>
  ) {
    return true;
  }

  template<unsigned long I, unsigned long... Is>
  constexpr static bool check_tuple_indices_impl(
    std::index_sequence<I, Is...>
  ) {
    return std::tuple_element<I, T>::type::i == I
      && TupleIndicesChecker<T>::check_tuple_indices_impl(std::index_sequence<Is...>());
  }
};

template<typename T>
constexpr bool check_tuple_indices() {
  using indices = typename tuple_indices<T>::type;
  return TupleIndicesChecker<T>::check_tuple_indices_impl(indices());
}
