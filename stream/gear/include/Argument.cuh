#pragma once

#include <tuple>

/**
 * @brief Macro for defining arguments in a Handler.
 */
#define ARGUMENTS(...) std::tuple<__VA_ARGS__>

/**
 * @brief Macro for defining arguments. An argument has an identifier
 *        and a type.
 */
#define ARGUMENT(ARGUMENT_NAME, ARGUMENT_TYPE)   \
  struct ARGUMENT_NAME {                         \
    constexpr static auto name {#ARGUMENT_NAME}; \
    using type = ARGUMENT_TYPE;                  \
    size_t size;                                 \
    char* offset;                                \
  };

/**
 * @brief Defines dependencies for an algorithm.
 *
 * @tparam T The algorithm type.
 * @tparam Args The dependencies.
 */
template<typename T, typename ArgumentsTuple>
struct AlgorithmDependencies {
  using Algorithm = T;
  using Arguments = ArgumentsTuple;
};

/**
 * @brief Dependencies for an algorithm, after
 *        being processed by the scheduler machinery.
 */
template<typename T, typename ArgumentsTuple>
struct ScheduledDependencies {
  using Algorithm = T;
  using Arguments = ArgumentsTuple;
};
