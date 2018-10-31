#include <iostream>
#include <tuple>
#include <functional>
#include <type_traits>
#include "Argument.cuh"

namespace Sch {

// Motivation:
// 
// I need somehow a struct with the inputs (things to malloc),
// and another one for the outputs (things to free), like so:
// 
// typedef std::tuple<
//   In<a_t, dev_a, dev_b>,
//   In<b_t, dev_c, dev_d>,
//   In<c_t>
// > input_t;
// 
// typedef std::tuple<
//   Out<a_t>,
//   Out<b_t, dev_a>,
//   Out<c_t, dev_c>,
//   Out<last_t, dev_b, dev_d>
// > output_t;

// A dummy for last element in Out
ALGORITHM(last_t)

// Checks whether an argument T is in any of the arguments specified in the Algorithms
template<typename T, typename Algorithms>
struct is_in_algorithms_arguments;

template<typename T>
struct is_in_algorithms_arguments<T, std::tuple<>> : std::false_type {};

template<typename T>
struct is_in_algorithms_arguments<T, std::tuple<last_t>> : std::false_type {};

template<typename T, typename A, typename... Args, typename... Algorithms>
struct is_in_algorithms_arguments<T, std::tuple<AlgorithmDependencies<A, Args...>, Algorithms...>> :
  std::conditional_t<tuple_contains<T, std::tuple<Args...>>::value, std::true_type,
    is_in_algorithms_arguments<T, std::tuple<Algorithms...>>>
  {};

// Finds algorithm in dependencies, and returns said dependencies
template<typename Algorithm, typename AlgorithmDeps>
struct FindAlgorithmDependencies;

template<typename Algorithm, typename OtherAlgorithm, typename... Arguments, typename... AlgorithmDeps>
struct FindAlgorithmDependencies<Algorithm, std::tuple<AlgorithmDependencies<OtherAlgorithm, Arguments...>, AlgorithmDeps...>> {
  using t = typename FindAlgorithmDependencies<Algorithm, std::tuple<AlgorithmDeps...>>::t;
};

template<typename Algorithm, typename... Arguments, typename... AlgorithmDeps>
struct FindAlgorithmDependencies<Algorithm, std::tuple<AlgorithmDependencies<Algorithm, Arguments...>, AlgorithmDeps...>> {
  using t = AlgorithmDependencies<Algorithm, Arguments...>;
};

// Keep only those dependencies that have been configured
// in configured_sequence_t
template<typename ConfiguredSequence, typename AlgorithmsDependencies>
struct ActiveSequence;

template<typename AlgorithmDeps>
struct ActiveSequence<std::tuple<>, AlgorithmDeps> {
  using t = std::tuple<>;
};

template<typename Algorithm, typename... Algorithms, typename AlgorithmDeps>
struct ActiveSequence<std::tuple<Algorithm, Algorithms...>, AlgorithmDeps> {
  // If some dependencies are not found, it should not compile.
  // Dependencies (even if empty) are required.
  using dependencies_for_algorithm = typename FindAlgorithmDependencies<Algorithm, AlgorithmDeps>::t;
  using t = typename tuple_append<
    typename ActiveSequence<std::tuple<Algorithms...>, AlgorithmDeps>::t,
    dependencies_for_algorithm
  >::t;
};

// A mechanism to only return the parameters in Algorithm
// that are not on any of the other Algorithms
template<typename Algorithm, typename Algorithms>
struct only_unused_types;

// If there are no other algorithms, return all the types
template<typename Algorithm, typename... Args>
struct only_unused_types<AlgorithmDependencies<Algorithm, Args...>, std::tuple<>> {
  using t = std::tuple<Args...>;
};

// Weird case: No dependencies in algo
template<typename Algorithm,
  typename... Algorithms>
struct only_unused_types<AlgorithmDependencies<Algorithm>, std::tuple<Algorithms...>> {
  using t = std::tuple<>;
};

template<typename Algorithm,
  typename Arg,
  typename... Args,
  typename AnotherAlgorithm,
  typename... Algorithms>
struct only_unused_types<AlgorithmDependencies<Algorithm, Arg, Args...>, std::tuple<AnotherAlgorithm, Algorithms...>> {
  // Types unused from Args...
  using previous_t = typename only_unused_types<AlgorithmDependencies<Algorithm, Args...>, std::tuple<AnotherAlgorithm, Algorithms...>>::t;

  // We append Arg only if it is _not_ on the previous algorithms
  using t = typename std::conditional_t<is_in_algorithms_arguments<Arg, std::tuple<AnotherAlgorithm, Algorithms...>>::value,
    previous_t,
    typename tuple_append<previous_t, Arg>::t>;
};

// Consume the algorithms and put their dependencies one by one
template<typename OutputArguments, typename Algorithms>
struct out_dependencies_impl;

template<typename OutputArguments, typename Algorithm, typename... Arguments>
struct out_dependencies_impl<OutputArguments, std::tuple<AlgorithmDependencies<Algorithm, Arguments...>, last_t>> {
  using t = std::tuple<AlgorithmDependencies<last_t, std::tuple<Arguments...>>>;
};

template<typename OutputArguments, typename Algorithm, typename... Arguments, typename NextAlgorithm, typename... NextAlgorithmArguments, typename... Algorithms>
struct out_dependencies_impl<OutputArguments, std::tuple<AlgorithmDependencies<Algorithm, Arguments...>,
    AlgorithmDependencies<NextAlgorithm, NextAlgorithmArguments...>, Algorithms...>> {

  using previous_t = typename out_dependencies_impl<OutputArguments, std::tuple<AlgorithmDependencies<NextAlgorithm, NextAlgorithmArguments...>, Algorithms...>>::t;
  using t = typename tuple_append<previous_t,
    AlgorithmDependencies<NextAlgorithm,
      typename tuple_elements_not_in<
        typename only_unused_types<AlgorithmDependencies<Algorithm, Arguments...>, std::tuple<AlgorithmDependencies<NextAlgorithm, NextAlgorithmArguments...>, Algorithms...>>::t,
        OutputArguments
      >::t
    >
  >::t;
};

// Helper to calculate OUT dependencies
template<typename ConfiguredSequence, typename OutputArguments, typename AlgorithmsDeps>
struct out_dependencies;

template<
  typename FirstAlgorithmInSequence,
  typename... RestOfSequence,
  typename OutputArguments,
  typename AlgorithmsDeps>
struct out_dependencies<
  std::tuple<FirstAlgorithmInSequence, RestOfSequence...>,
  OutputArguments,
  AlgorithmsDeps> {
  using t =
    typename tuple_reverse<
      typename tuple_append<
        typename out_dependencies_impl<OutputArguments,
          typename tuple_append<
            typename ActiveSequence<
              typename tuple_reverse<std::tuple<FirstAlgorithmInSequence, RestOfSequence...>>::t, AlgorithmsDeps
            >::t,
            last_t
          >::t
        >::t,
        AlgorithmDependencies<FirstAlgorithmInSequence, std::tuple<>>
      >::t
    >::t;
};

template<typename ConfiguredSequence, typename OutputArguments>
struct out_dependencies<ConfiguredSequence, OutputArguments, std::tuple<>> {
  using t = std::tuple<AlgorithmDependencies<last_t, std::tuple<>>>;
};

// Consume the algorithms and put their dependencies one by one
template<typename AlgorithmsDeps>
struct in_dependencies_impl;

template<>
struct in_dependencies_impl<std::tuple<>> {
  using t = std::tuple<>;
};

template<typename Algorithm, typename... Arguments, typename... AlgorithmsDeps>
struct in_dependencies_impl<std::tuple<AlgorithmDependencies<Algorithm, Arguments...>, AlgorithmsDeps...>> {
  using previous_t = typename in_dependencies_impl<std::tuple<AlgorithmsDeps...>>::t;
  using t = typename tuple_append<previous_t,
    AlgorithmDependencies<Algorithm, typename only_unused_types<AlgorithmDependencies<Algorithm, Arguments...>, std::tuple<AlgorithmsDeps...>>::t>>::t;
};

template<typename ConfiguredSequence, typename AlgorithmsDeps>
using in_dependencies = in_dependencies_impl<typename tuple_reverse<
    typename ActiveSequence<typename tuple_reverse<ConfiguredSequence>::t, AlgorithmsDeps>::t
  >::t>;

// Fetches all arguments from ie. in_dependencies into a tuple
template<typename in_deps>
struct ArgumentsTuple;

template<>
struct ArgumentsTuple<std::tuple<>> {
  using t = std::tuple<>;
};

template<typename Algorithm, typename... Algorithms>
struct ArgumentsTuple<std::tuple<AlgorithmDependencies<Algorithm, std::tuple<>>, Algorithms...>> {
  using t = typename ArgumentsTuple<std::tuple<Algorithms...>>::t;
};

template<typename Algorithm, typename Arg, typename... Args, typename... Algorithms>
struct ArgumentsTuple<std::tuple<AlgorithmDependencies<Algorithm, std::tuple<Arg, Args...>>, Algorithms...>> {
  using previous_t = typename ArgumentsTuple<std::tuple<AlgorithmDependencies<Algorithm, std::tuple<Args...>>, Algorithms...>>::t;
  using t = typename tuple_append<previous_t, Arg>::t;
};

// Helper to just print the arguments
template<typename Arguments>
struct print_arguments;

template<>
struct print_arguments<std::tuple<>> {
  static constexpr void print() {}
};

template<typename Argument, typename... Arguments>
struct print_arguments<std::tuple<Argument, Arguments...>> {
  static constexpr void print() {
    std::cout << Argument::name << ", ";
    print_arguments<std::tuple<Arguments...>>::print();
  }
};

// Iterate the types (In or Out) and print them for each iteration
template<typename Dependencies>
struct print_algorithm_dependencies;

template<>
struct print_algorithm_dependencies<std::tuple<>> {
  static constexpr void print() {};
};

template<typename Algorithm, typename... Arguments, typename... Dependencies>
struct print_algorithm_dependencies<std::tuple<AlgorithmDependencies<Algorithm, Arguments...>, Dependencies...>> {
  static constexpr void print() {
    std::cout << "Algorithm " << Algorithm::name << ":" << std::endl
      << std::tuple_size<Arguments...>::value << " dependencies" << std::endl;

    print_arguments<Arguments...>::print();
    std::cout << std::endl << std::endl;

    print_algorithm_dependencies<std::tuple<Dependencies...>>::print();
  }
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

}
