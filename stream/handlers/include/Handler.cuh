#pragma once

#include "cuda_runtime.h"
#include <ostream>
#include <tuple>
#include <utility>

template<typename Tuple>
struct tuple_indices {
  using type = decltype(std::make_index_sequence<std::tuple_size<Tuple>::value>());
};

template<class Fn, class Tuple, unsigned long... I>
auto call_impl(
  Fn fn,
  const dim3& num_blocks,
  const dim3& num_threads,
  const unsigned shared_memory_size,
  cudaStream_t* stream,
  Tuple&& t,
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
  Tuple args
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

template<typename R, typename... T>
struct Handler {
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

  void operator()() {
    call(function, num_blocks, num_threads,
      shared_memory_size, stream, arguments);
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
};

template<typename R, typename... T>
Handler<R, T...> generate_handler(R(f)(T...)) {
  return Handler<R, T...>{f};
}
