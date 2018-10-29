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

// Appends a Tuple with the Element
template<typename Tuple, typename Element>
struct tuple_append;

template<typename... T, typename E>
struct tuple_append<std::tuple<T...>, E> {
  using t = std::tuple<T..., E>;
};

template<typename E>
struct tuple_append<std::tuple<>, E> {
  using t = std::tuple<E>;
};

// Reverses a tuple
template<typename Tuple>
struct tuple_reverse;

template<>
struct tuple_reverse<std::tuple<>> {
  using t = std::tuple<>;
};

template<typename T, typename... Elements>
struct tuple_reverse<std::tuple<T, Elements...>> {
  using previous_t = typename tuple_reverse<std::tuple<Elements...>>::t;
  using t = typename tuple_append<previous_t, T>::t;
};

// Returns types in Tuple not in OtherTuple
template<typename Tuple, typename OtherTuple>
struct tuple_elements_not_in;

template<typename OtherTuple>
struct tuple_elements_not_in<std::tuple<>, OtherTuple> {
  using t = std::tuple<>;
};

template<typename Tuple>
struct tuple_elements_not_in<Tuple, std::tuple<>> {
  using t = Tuple;
};

template<typename T, typename... Elements, typename OtherTuple>
struct tuple_elements_not_in<std::tuple<T, Elements...>, OtherTuple> {
  using previous_t = typename tuple_elements_not_in<std::tuple<Elements...>, OtherTuple>::t;
  using t = typename std::conditional_t<tuple_contains<T, OtherTuple>::value,
    previous_t,
    typename tuple_append<previous_t, T>::t>;
};
