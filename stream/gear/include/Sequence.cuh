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

  /**
   * @brief Fetches an item of the sequence.
   */
  template<unsigned long I>
  decltype(std::get<I>(algorithms)) item() noexcept {
    return std::get<I>(algorithms);
  }

  /**
   * @brief Wraps set_arguments of the element in the sequence.
   */
  template<unsigned long I, typename... A>
  void set_arguments(A... arguments) {
    std::get<I>(algorithms).set_arguments(arguments...);
  }

  /**
   * @brief Wraps set_opts of the element in the sequence.
   */
  template<unsigned long I>
  void set_opts(
    const dim3& param_num_blocks,
    const dim3& param_num_threads,
    cudaStream_t& param_stream,
    const unsigned param_shared_memory_size = 0
  ) {
    std::get<I>(algorithms).set_opts(param_num_blocks, param_num_threads,
      param_stream, param_shared_memory_size);
  }

  /**
   * @brief Wraps invoke of the element in the sequence.
   */
  template<unsigned long I>
  void invoke() {
    std::get<I>(algorithms).invoke();
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
