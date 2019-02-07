#pragma once

#include "ArgumentManager.cuh"
#include <functional>
#include <tuple>
#include <utility>

/**
 * @brief      Macro for defining CPU algorithms defined by a function name.
 *             A struct is created with name EXPOSED_TYPE_NAME that encapsulates
 *             a CpuHandler of type FUNCTION_NAME.
 */
#define CPU_ALGORITHM(FUNCTION_NAME, EXPOSED_TYPE_NAME, DEPENDENCIES)         \
  struct EXPOSED_TYPE_NAME {                                                  \
    constexpr static auto name {#EXPOSED_TYPE_NAME};                          \
    using Arguments = DEPENDENCIES;                                           \
    using arguments_t = ArgumentRefManager<Arguments>;                        \
    decltype(make_cpu_handler(FUNCTION_NAME)) handler {FUNCTION_NAME};        \
    template<typename... T>                                                   \
    auto invoke(T&&... arguments)                                             \
    {                                                                         \
      return handler.function(std::forward<T>(arguments)...);                 \
    }                                                                         \
  };

/**
 * @brief      A Handler that encapsulates a CPU function.
 *             set_arguments allows to set up the arguments of the function.
 *             invokes calls it.
 */
template<typename R, typename... T>
struct CpuHandler {
  std::function<R(T...)> function;
  CpuHandler(std::function<R(T...)> param_function) : function(param_function) {}
};

/**
 * @brief      A helper to make Handlers without needing
 *             to specify its function type (ie. "make_cpu_handler(function)").
 */
template<typename R, typename... T>
static CpuHandler<R, T...> make_cpu_handler(R(f)(T...))
{
  return CpuHandler<R, T...> {f};
}
