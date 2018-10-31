#pragma once

#include <tuple>

/**
 * @brief Macro for defining arguments. An argument has an identifier
 *        and a type.
 */
#define ARGUMENT(ARGUMENT_NAME, ARGUMENT_TYPE) \
  struct ARGUMENT_NAME {\
    constexpr static auto name {#ARGUMENT_NAME};\
    using type = ARGUMENT_TYPE;\
  };

/**
 * @brief Defines dependencies for an algorithm.
 * 
 * @tparam T The algorithm type.
 * @tparam Args The dependencies.
 */
template<typename T, typename... Args>
struct AlgorithmDependencies {
  using Algorithm = T;
  using Arguments = std::tuple<Args...>;
};
