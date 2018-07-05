#pragma once

#include <tuple>

template<typename Tuple>
struct tuple_indices {
  using type = decltype(std::make_index_sequence<std::tuple_size<Tuple>::value>());
};
