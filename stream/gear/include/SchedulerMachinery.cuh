#pragma once

#include <iostream>
#include <tuple>
#include <functional>
#include <type_traits>
#include "Argument.cuh"
#include "TupleTools.cuh"

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
struct last_t {};

// Checks whether an argument T is in any of the arguments specified in the Algorithms
template<typename T, typename Algorithms>
struct IsInAlgorithmsArguments;

template<typename T>
struct IsInAlgorithmsArguments<T, std::tuple<>> : std::false_type {};

template<typename T>
struct IsInAlgorithmsArguments<T, std::tuple<last_t>> : std::false_type {};

template<typename T, typename A, typename... Args, typename... Algorithms>
struct IsInAlgorithmsArguments<T, std::tuple<AlgorithmDependencies<A, Args...>, Algorithms...>> :
  std::conditional_t<TupleContains<T, std::tuple<Args...>>::value, std::true_type,
    IsInAlgorithmsArguments<T, std::tuple<Algorithms...>>>
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
  using t = typename TupleAppend<
    typename ActiveSequence<std::tuple<Algorithms...>, AlgorithmDeps>::t,
    dependencies_for_algorithm
  >::t;
};

// A mechanism to only return the arguments in Algorithm
// that are not on any of the other Algorithms
template<typename Algorithm, typename Algorithms>
struct ArgumentsNotIn;

// If there are no other algorithms, return all the types
template<typename Algorithm, typename... Args>
struct ArgumentsNotIn<AlgorithmDependencies<Algorithm, Args...>, std::tuple<>> {
  using t = std::tuple<Args...>;
};

// Weird case: No dependencies in algo
template<typename Algorithm,
  typename... Algorithms>
struct ArgumentsNotIn<AlgorithmDependencies<Algorithm>, std::tuple<Algorithms...>> {
  using t = std::tuple<>;
};

template<typename Algorithm,
  typename Arg,
  typename... Args,
  typename AnotherAlgorithm,
  typename... Algorithms>
struct ArgumentsNotIn<AlgorithmDependencies<Algorithm, Arg, Args...>, std::tuple<AnotherAlgorithm, Algorithms...>> {
  // Types unused from Args...
  using previous_t = typename ArgumentsNotIn<AlgorithmDependencies<Algorithm, Args...>, std::tuple<AnotherAlgorithm, Algorithms...>>::t;

  // We append Arg only if it is _not_ on the previous algorithms
  using t = typename std::conditional_t<IsInAlgorithmsArguments<Arg, std::tuple<AnotherAlgorithm, Algorithms...>>::value,
    previous_t,
    typename TupleAppend<previous_t, Arg>::t>;
};

// Consume the algorithms and put their dependencies one by one
template<typename OutputArguments, typename Algorithms>
struct OutDependenciesImpl;

template<typename OutputArguments, typename Algorithm, typename... Arguments>
struct OutDependenciesImpl<OutputArguments, std::tuple<AlgorithmDependencies<Algorithm, Arguments...>, last_t>> {
  using t = std::tuple<AlgorithmDependencies<last_t, typename TupleElementsNotIn<
    std::tuple<Arguments...>,
    OutputArguments>::t
  >>;
};

template<typename OutputArguments, typename Algorithm, typename... Arguments, typename NextAlgorithm, typename... NextAlgorithmArguments, typename... Algorithms>
struct OutDependenciesImpl<OutputArguments, std::tuple<AlgorithmDependencies<Algorithm, Arguments...>,
    AlgorithmDependencies<NextAlgorithm, NextAlgorithmArguments...>, Algorithms...>> {

  using previous_t = typename OutDependenciesImpl<OutputArguments, std::tuple<AlgorithmDependencies<NextAlgorithm, NextAlgorithmArguments...>, Algorithms...>>::t;
  using t = typename TupleAppend<previous_t,
    ScheduledDependencies<NextAlgorithm,
      typename TupleElementsNotIn<
        typename ArgumentsNotIn<AlgorithmDependencies<Algorithm, Arguments...>, std::tuple<AlgorithmDependencies<NextAlgorithm, NextAlgorithmArguments...>, Algorithms...>>::t,
        OutputArguments
      >::t
    >
  >::t;
};

// Helper to calculate OUT dependencies
template<typename ConfiguredSequence, typename OutputArguments, typename AlgorithmsDeps>
struct OutDependencies;

template<
  typename FirstAlgorithmInSequence,
  typename... RestOfSequence,
  typename OutputArguments,
  typename AlgorithmsDeps>
struct OutDependencies<
  std::tuple<FirstAlgorithmInSequence, RestOfSequence...>,
  OutputArguments,
  AlgorithmsDeps> {
  using t =
    typename TupleReverse<
      typename TupleAppend<
        typename OutDependenciesImpl<OutputArguments,
          typename TupleAppend<
            typename ActiveSequence<
              typename TupleReverse<std::tuple<FirstAlgorithmInSequence, RestOfSequence...>>::t, AlgorithmsDeps
            >::t,
            last_t
          >::t
        >::t,
        ScheduledDependencies<FirstAlgorithmInSequence, std::tuple<>>
      >::t
    >::t;
};

template<typename ConfiguredSequence, typename OutputArguments>
struct OutDependencies<ConfiguredSequence, OutputArguments, std::tuple<>> {
  using t = std::tuple<ScheduledDependencies<last_t, std::tuple<>>>;
};

// Consume the algorithms and put their dependencies one by one
template<typename AlgorithmsDeps>
struct InDependenciesImpl;

template<>
struct InDependenciesImpl<std::tuple<>> {
  using t = std::tuple<>;
};

template<typename Algorithm, typename... Arguments, typename... AlgorithmsDeps>
struct InDependenciesImpl<std::tuple<AlgorithmDependencies<Algorithm, Arguments...>, AlgorithmsDeps...>> {
  using previous_t = typename InDependenciesImpl<std::tuple<AlgorithmsDeps...>>::t;
  using t = typename TupleAppend<previous_t,
    ScheduledDependencies<Algorithm, typename ArgumentsNotIn<AlgorithmDependencies<Algorithm, Arguments...>, std::tuple<AlgorithmsDeps...>>::t>>::t;
};

template<typename ConfiguredSequence, typename AlgorithmsDeps>
using InDependencies =
  InDependenciesImpl<
    typename TupleReverse<
      typename ActiveSequence<
        typename TupleReverse<ConfiguredSequence>::t, AlgorithmsDeps
      >::t
    >::t
  >;

// Fetches all arguments from ie. InDependencies into a tuple
template<typename in_deps>
struct ArgumentsTuple;

template<>
struct ArgumentsTuple<std::tuple<>> {
  using t = std::tuple<>;
};

template<typename Algorithm, typename... Algorithms>
struct ArgumentsTuple<std::tuple<ScheduledDependencies<Algorithm, std::tuple<>>, Algorithms...>> {
  using t = typename ArgumentsTuple<std::tuple<Algorithms...>>::t;
};

template<typename Algorithm, typename Arg, typename... Args, typename... Algorithms>
struct ArgumentsTuple<std::tuple<ScheduledDependencies<Algorithm, std::tuple<Arg, Args...>>, Algorithms...>> {
  using previous_t = typename ArgumentsTuple<std::tuple<ScheduledDependencies<Algorithm, std::tuple<Args...>>, Algorithms...>>::t;
  using t = typename TupleAppend<previous_t, Arg>::t;
};

// Helper to just print the arguments
template<typename Arguments>
struct PrintArguments;

template<>
struct PrintArguments<std::tuple<>> {
  static constexpr void print() {}
};

template<typename Argument, typename... Arguments>
struct PrintArguments<std::tuple<Argument, Arguments...>> {
  static constexpr void print() {
    std::cout << Argument::name << ", ";
    PrintArguments<std::tuple<Arguments...>>::print();
  }
};

// Iterate the types (In or Out) and print them for each iteration
template<typename Dependencies>
struct PrintAlgorithmDependencies;

template<>
struct PrintAlgorithmDependencies<std::tuple<>> {
  static constexpr void print() {};
};

template<typename Algorithm, typename... Arguments, typename... Dependencies>
struct PrintAlgorithmDependencies<std::tuple<ScheduledDependencies<Algorithm, std::tuple<Arguments...>>, Dependencies...>> {
  static constexpr void print() {
    std::cout << "Algorithm " << Algorithm::name << ":" << std::endl
      << std::tuple_size<std::tuple<Arguments...>>::value << " dependencies" << std::endl;

    PrintArguments<Arguments...>::print();
    std::cout << std::endl << std::endl;

    PrintAlgorithmDependencies<std::tuple<Dependencies...>>::print();
  }
};

/**
 * @brief Runs the sequence tuple (implementation).
 */
template<typename Scheduler,
  typename Functor,
  typename Tuple,
  typename SetSizeArguments,
  typename VisitArguments,
  typename Indices>
struct RunSequenceTupleImpl;

template<typename Scheduler,
  typename Functor,
  typename Tuple,
  typename... SetSizeArguments,
  typename... VisitArguments>
struct RunSequenceTupleImpl<Scheduler, Functor, Tuple, std::tuple<SetSizeArguments...>, std::tuple<VisitArguments...>, std::index_sequence<>> {
  constexpr static void run(
    Scheduler& scheduler,
    Functor& functor,
    Tuple& tuple,
    SetSizeArguments&&... set_size_arguments,
    VisitArguments&&... visit_arguments) {}
};

template<typename Scheduler,
  typename Functor,
  typename Tuple,
  typename... SetSizeArguments,
  typename... VisitArguments,
  unsigned long I,
  unsigned long... Is>
struct RunSequenceTupleImpl<Scheduler, Functor, Tuple, std::tuple<SetSizeArguments...>, std::tuple<VisitArguments...>, std::index_sequence<I, Is...>> {
  constexpr static void run(
    Scheduler& scheduler,
    Functor& functor,
    Tuple& tuple,
    SetSizeArguments&&... set_size_arguments,
    VisitArguments&&... visit_arguments)
  {
    using t = typename std::tuple_element<I, Tuple>::type;

    // Sets the arguments sizes, setups the scheduler and visits the algorithm.
    functor.template set_arguments_size<t>(std::forward<SetSizeArguments>(set_size_arguments)...);
    scheduler.template setup<I, t>();
    functor.template visit<t>(std::get<I>(tuple), std::forward<VisitArguments>(visit_arguments)...);

    RunSequenceTupleImpl<
      Scheduler,
      Functor,
      Tuple,
      std::tuple<SetSizeArguments...>,
      std::tuple<VisitArguments...>,
      std::index_sequence<Is...>>::run(scheduler, functor, tuple, std::forward<SetSizeArguments>(set_size_arguments)..., std::forward<VisitArguments>(visit_arguments)...);
  }
};

/**
 * @brief Runs a sequence of algorithms.
 * 
 * @tparam Functor          Functor containing the visit function for all types in the tuple.
 * @tparam Tuple            Sequence of algorithms
 * @tparam SetSizeArguments Arguments to set_arguments_size
 * @tparam VisitArguments   Arguments to visit
 */
template<typename Scheduler, typename Functor, typename Tuple, typename SetSizeArguments, typename VisitArguments>
struct RunSequenceTuple;

template<typename Scheduler, typename Functor, typename Tuple, typename... SetSizeArguments, typename... VisitArguments>
struct RunSequenceTuple<Scheduler, Functor, Tuple, std::tuple<SetSizeArguments...>, std::tuple<VisitArguments...>> {
  constexpr static void run(
    Scheduler& scheduler,
    Functor& functor,
    Tuple& tuple,
    SetSizeArguments&&... set_size_arguments,
    VisitArguments&&... visit_arguments)
  {
    RunSequenceTupleImpl<
      Scheduler,
      Functor,
      Tuple,
      std::tuple<SetSizeArguments...>,
      std::tuple<VisitArguments...>,
      std::make_index_sequence<std::tuple_size<Tuple>::value>
    >::run(scheduler, functor, tuple, std::forward<SetSizeArguments>(set_size_arguments)..., std::forward<VisitArguments>(visit_arguments)...);
  }
};

}
