#pragma once

#include <tuple>
#include <functional>

template<typename Tuple>
struct tuple_indices {
  using type = decltype(std::make_index_sequence<std::tuple_size<Tuple>::value>());
};

template <typename Ftor, typename Tuple, size_t... Is, typename... Args>
void apply_unary_impl(Ftor&& ftor, Tuple&& tuple, std::index_sequence<Is...>, Args&&... args) {
    auto _ = { (ftor(std::get<Is>(std::forward<Tuple>(tuple)), args...), void(), 0)... };
}

template <typename Ftor, typename Tuple, typename... Args>
void apply_unary(Ftor&& ftor, Tuple&& tuple, Args&&... args) {
    apply_unary_impl(std::forward<Ftor>(ftor),
                     std::forward<Tuple>(tuple),
                     std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value> {},
                     args...);
}

template <class T, class Tuple>
struct tuple_index;

template <class T, class... Types>
struct tuple_index<T, std::tuple<T, Types...>> {
  static const std::size_t value = 0;
};

template <class T, class U, class... Types>
struct tuple_index<T, std::tuple<U, Types...>> {
  static const std::size_t value = 1 + tuple_index<T, std::tuple<Types...>>::value;
};
