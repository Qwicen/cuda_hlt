#pragma once

#include <tuple>
#include <functional>
#include <type_traits>

/**
 * @brief Checks if a tuple contains type T, and obtains index.
 *        index will be the length of the tuple if the type was not found.
 *
 *        Some examples of its usage:
 *
 *        if (TupleContains<int, decltype(t)>::value) {
 *          std::cout << "t contains int" << std::endl;
 *        }
 *
 *        std::cout << "int in index " << TupleContains<int, decltype(t)>::index << std::endl;
 */
template<typename T, typename Tuple>
struct TupleContains;

template<typename T>
struct TupleContains<T, std::tuple<>> : std::false_type {
  static constexpr int index = 0;
};

template<typename T, typename... Ts>
struct TupleContains<T, std::tuple<T, Ts...>> : std::true_type {
  static constexpr int index = 0;
};

template<typename T, typename U, typename... Ts>
struct TupleContains<T, std::tuple<U, Ts...>> : TupleContains<T, std::tuple<Ts...>> {
  static constexpr int index = 1 + TupleContains<T, std::tuple<Ts...>>::index;
};

// Appends a Tuple with the Element
template<typename Tuple, typename Element>
struct TupleAppend;

template<typename... T, typename E>
struct TupleAppend<std::tuple<T...>, E> {
  using t = std::tuple<T..., E>;
};

template<typename E>
struct TupleAppend<std::tuple<>, E> {
  using t = std::tuple<E>;
};

// Appends a Tuple with the Element
template<typename Tuple, typename Element>
struct TupleAppendFirst;

template<typename E, typename... T>
struct TupleAppendFirst<E, std::tuple<T...>> {
  using t = std::tuple<E, T...>;
};

template<typename E>
struct TupleAppendFirst<E, std::tuple<>> {
  using t = std::tuple<E>;
};

// Reverses a tuple
template<typename Tuple>
struct TupleReverse;

template<>
struct TupleReverse<std::tuple<>> {
  using t = std::tuple<>;
};

template<typename T, typename... Elements>
struct TupleReverse<std::tuple<T, Elements...>> {
  using previous_t = typename TupleReverse<std::tuple<Elements...>>::t;
  using t = typename TupleAppend<previous_t, T>::t;
};

// Returns types in Tuple not in OtherTuple
template<typename Tuple, typename OtherTuple>
struct TupleElementsNotIn;

template<typename OtherTuple>
struct TupleElementsNotIn<std::tuple<>, OtherTuple> {
  using t = std::tuple<>;
};

template<typename Tuple>
struct TupleElementsNotIn<Tuple, std::tuple<>> {
  using t = Tuple;
};

template<typename T, typename... Elements, typename OtherTuple>
struct TupleElementsNotIn<std::tuple<T, Elements...>, OtherTuple> {
  using previous_t = typename TupleElementsNotIn<std::tuple<Elements...>, OtherTuple>::t;
  using t = typename std::
    conditional_t<TupleContains<T, OtherTuple>::value, previous_t, typename TupleAppend<previous_t, T>::t>;
};
