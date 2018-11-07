#pragma once

#include <tuple>
#include <functional>
#include <type_traits>

/**
 * @brief      Utility to fetch tuple indices.
 */
template<typename Tuple>
struct tuple_indices {
  using type = decltype(std::make_index_sequence<std::tuple_size<Tuple>::value>());
};

/**
 * @brief Checks if a tuple contains type T, and obtains index.
 *        index will be the length of the tuple if the type was not found.
 *
 *        Some examples of its usage:
 *        
 *        if (tuple_contains<int, decltype(t)>::value) {
 *          std::cout << "t contains int" << std::endl;
 *        }
 *        
 *        std::cout << "int in index " << tuple_contains<int, decltype(t)>::index << std::endl;
 */
template <typename T, typename Tuple>
struct tuple_contains;

template <typename T>
struct tuple_contains<T, std::tuple<>> : std::false_type {
  static constexpr int index = 0;
};

template <typename T, typename... Ts>
struct tuple_contains<T, std::tuple<T, Ts...>> : std::true_type {
  static constexpr int index = 0;
};

template <typename T, typename U, typename... Ts>
struct tuple_contains<T, std::tuple<U, Ts...>> : tuple_contains<T, std::tuple<Ts...>> {
  static constexpr int index = 1 + tuple_contains<T, std::tuple<Ts...>>::index;
};

/**
 * @brief      Runs a sequence of algorithms (implementation).
 *
 * @param[in]  ftor       Functor containing the visit function for all types in the tuple.
 * @param[in]  tuple      Sequence of algorithms.
 * @param[in]  Is         Indices of all elements in the tuple.
 * @param[in]  args       Additional arguments to the visit function.
 */
template <typename Ftor, typename Tuple, size_t... Is, typename... Args>
void run_sequence_tuple_impl(Ftor&& ftor, Tuple&& tuple, std::index_sequence<Is...>, Args&&... args) {
    auto _ = { (ftor.visit(std::get<Is>(std::forward<Tuple>(tuple)), Is, args...), void(), 0)... };
}

/**
 * @brief      Runs a sequence of algorithms.
 *
 * @param[in]  ftor       Functor containing the visit function for all types in the tuple.
 * @param[in]  tuple      Sequence of algorithms.
 * @param[in]  args       Additional arguments to the visit function.
 */
template <typename Ftor, typename Tuple, typename... Args>
void run_sequence_tuple(Ftor&& ftor, Tuple&& tuple, Args&&... args) {
    run_sequence_tuple_impl(std::forward<Ftor>(ftor),
                     std::forward<Tuple>(tuple),
                     std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value> {},
                     args...);
}
