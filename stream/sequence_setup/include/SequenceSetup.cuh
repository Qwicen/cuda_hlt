#pragma once

#include <iostream>
#include "ConfiguredSequence.cuh"
#include "TupleIndicesChecker.cuh"
#include "Arguments.cuh"

// Prepared for C++17 variant
// template<typename T>
// auto transition(const T& state) {
//   return typename std::tuple_element<tuple_index<T, sequence_tuple_n>::value + 1, sequence_tuple_n>::type{};
// }

/**
 * @brief Retrieves the sequence dependencies.
 * @details The sequence dependencies specifies for each algorithm
 *          in the sequence the datatypes it depends on from the arguments.
 *
 *          Note that this vector of arguments may vary from the actual
 *          arguments in the kernel invocation: ie. some cases:
 *          * if something is passed by value
 *          * if a pointer is set to point somewhere different from the beginning
 *          * if an argument is repeated in the argument list.
 */
std::vector<std::vector<int>> get_sequence_dependencies();

/**
 * @brief Retrieves the persistent datatypes.
 * @details The sequence may contain some datatypes that
 *          once they are reserved should never go out of memory (persistent).
 *          ie. the tracking sequence may produce some result and
 *          the primary vertex recostruction a different result.
 *          All output arguments should be returned here.
 */
std::vector<int> get_sequence_output_arguments();
