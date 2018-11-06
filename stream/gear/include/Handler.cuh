#pragma once

#include <ostream>
#include <tuple>
#include <utility>

/**
 * @brief      Macro for defining algorithms defined by a function name.
 *             A struct is created with name EXPOSED_TYPE_NAME that encapsulates
 *             a Handler of type FUNCTION_NAME.
 */
#define ALGORITHM(FUNCTION_NAME, EXPOSED_TYPE_NAME) \
  struct EXPOSED_TYPE_NAME {\
    constexpr static auto name {#EXPOSED_TYPE_NAME};\
    decltype(HandlerMaker::make_handler(FUNCTION_NAME)) handler {FUNCTION_NAME};\
    void set_opts(\
      const dim3& param_num_blocks,\
      const dim3& param_num_threads,\
      cudaStream_t& param_stream,\
      const unsigned param_shared_memory_size = 0\
    ) {\
      handler.set_opts(param_num_blocks, param_num_threads,\
        param_stream, param_shared_memory_size);\
    }\
    template<typename... T>\
    void set_arguments(T... param_arguments) {\
      handler.set_arguments(param_arguments...);\
    }\
    void invoke() {\
      handler.invoke();\
    }\
  };

/**
 * @brief      Invokes a function specified by its function and arguments.
 *
 * @param[in]  function            The function.
 * @param[in]  num_blocks          Number of blocks of kernel invocation.
 * @param[in]  num_threads         Number of threads of kernel invocation.
 * @param[in]  shared_memory_size  Shared memory size.
 * @param      stream              The stream where the function will be run.
 * @param[in]  arguments           The arguments of the function.
 * @param[in]  I                   Index sequence
 *
 * @return     Return value of the function.
 */
template<class Fn, class Tuple, unsigned long... I>
auto invoke_impl(
  Fn function,
  const dim3& num_blocks,
  const dim3& num_threads,
  const unsigned shared_memory_size,
  cudaStream_t* stream,
  const Tuple& arguments,
  std::index_sequence<I...>
) -> decltype(function(std::get<I>(arguments)...)) {
  return function<<<num_blocks, num_threads, shared_memory_size, *stream>>>(std::get<I>(arguments)...);
}

/**
 * @brief      A Handler that encapsulates a CUDA function.
 *             It exposes set_opts, to set its CUDA specific function
 *             call parameters (inside the <<< >>>).
 *             set_arguments allows to set up the arguments of the function.
 */
template<typename R, typename... T>
struct Handler {
  dim3 num_blocks, num_threads;
  unsigned shared_memory_size = 0;
  cudaStream_t* stream;

  // Call arguments and function
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
    invoke_impl(function, num_blocks, num_threads,
      shared_memory_size, stream, arguments,
      std::make_index_sequence<std::tuple_size<std::tuple<T...>>::value>());
  }
};

/**
 * @brief      A helper to make Handlers without needing
 *             to specify its function type (ie. "HandlerMaker::make_handler(function)").
 */
struct HandlerMaker {
  template<typename R, typename... T>
  static Handler<R, T...> make_handler(R(f)(T...)) {
    return Handler<R, T...>{f};
  }
};
