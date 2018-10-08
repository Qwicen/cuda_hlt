#pragma once

#include "cuda_runtime.h"
#include <ostream>
#include <tuple>
#include <utility>
#include "Common.cuh"

template<class Fn, class Tuple, unsigned long... I>
auto call_impl(
  Fn fn,
  const dim3& num_blocks,
  const dim3& num_threads,
  const unsigned shared_memory_size,
  cudaStream_t* stream,
  const Tuple& t,
  std::index_sequence<I...>
) -> decltype(fn(std::get<I>(t)...)) {
  return fn<<<num_blocks, num_threads, shared_memory_size, *stream>>>(std::get<I>(t)...);
}

template<typename Fn, typename Tuple>
auto call(
  Fn fn,
  const dim3& num_blocks,
  const dim3& num_threads,
  const unsigned shared_memory_size,
  cudaStream_t* stream,
  const Tuple& args
) {
  using indices = typename tuple_indices<Tuple>::type;
  return call_impl(fn, num_blocks, num_threads, shared_memory_size,
    stream, args, indices());
}

template<typename R, typename... T>
std::tuple<T...> tuple_parameters (R(_)(T...)) {
  return std::tuple<T...>{};
}

template<typename R, typename... T>
R return_type (R(_)(T...)) {
  return R{};
}

template<unsigned long I, typename R, typename... T>
struct Handler {
  constexpr static unsigned long i = I;
  dim3 num_blocks, num_threads;
  unsigned shared_memory_size = 0;
  cudaStream_t* stream;

  // Call arguments
  std::tuple<T...> arguments;
  R(*function)(T...);

  Handler() = default;

  Handler(R(*param_function)(T...)) : function(param_function) {}

  void set_arguments(T... param_arguments) {
    arguments = std::tuple<T...>{param_arguments...};
  }

  void set_opts(
    const dim3& param_num_blocks,
    const dim3& param_num_threads,
    cudaStream_t& param_stream,
    const unsigned param_shared_memory_size = 0
  ) {
    num_blocks = param_num_blocks;
    num_threads = param_num_threads;
    stream = &param_stream;
    shared_memory_size = param_shared_memory_size;
  }
  
  void invoke() {
    call(function, num_blocks, num_threads,
      shared_memory_size, stream, arguments);
  }
};
