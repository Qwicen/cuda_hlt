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
