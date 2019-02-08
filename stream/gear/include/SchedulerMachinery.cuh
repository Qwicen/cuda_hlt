#pragma once

#include <tuple>
#include <functional>
#include <type_traits>
#include "Logger.h"
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
  struct last_t {
    constexpr static auto name {"last"};
    using Arguments = std::tuple<>;
  };

  // Checks whether an argument T is in any of the arguments specified in the Algorithms
  template<typename T, typename Algorithms>
  struct IsInAlgorithmsArguments;

  template<typename T>
  struct IsInAlgorithmsArguments<T, std::tuple<>> : std::false_type {
  };

  template<typename T>
  struct IsInAlgorithmsArguments<T, std::tuple<last_t>> : std::false_type {
  };

  template<typename T, typename Algorithm, typename... Algorithms>
  struct IsInAlgorithmsArguments<T, std::tuple<Algorithm, Algorithms...>>
    : std::conditional_t<
        TupleContains<T, typename Algorithm::Arguments>::value,
        std::true_type,
        IsInAlgorithmsArguments<T, std::tuple<Algorithms...>>> {
  };

  // A mechanism to only return the arguments in Algorithm
  // that are not on any of the other Algorithms
  template<typename Arguments, typename Algorithms>
  struct ArgumentsNotIn;

  // If there are no other algorithms, return all the types
  template<typename... Args>
  struct ArgumentsNotIn<std::tuple<Args...>, std::tuple<>> {
    using t = std::tuple<Args...>;
  };

  // Weird case: No dependencies in algo
  template<typename... Algorithms>
  struct ArgumentsNotIn<std::tuple<>, std::tuple<Algorithms...>> {
    using t = std::tuple<>;
  };

  template<typename Arg, typename... Args, typename AnotherAlgorithm, typename... Algorithms>
  struct ArgumentsNotIn<std::tuple<Arg, Args...>, std::tuple<AnotherAlgorithm, Algorithms...>> {
    // Types unused from Args...
    using previous_t = typename ArgumentsNotIn<std::tuple<Args...>, std::tuple<AnotherAlgorithm, Algorithms...>>::t;

    // We append Arg only if it is _not_ on the previous algorithms
    using t = typename std::conditional_t<
      IsInAlgorithmsArguments<Arg, std::tuple<AnotherAlgorithm, Algorithms...>>::value,
      previous_t,
      typename TupleAppend<previous_t, Arg>::t>;
  };

  // Consume the algorithms and put their dependencies one by one
  template<typename OutputArguments, typename Algorithms>
  struct OutDependenciesImpl;

  template<typename OutputArguments, typename Algorithm>
  struct OutDependenciesImpl<OutputArguments, std::tuple<Algorithm, last_t>> {
    using t =
      std::tuple<last_t, std::tuple<typename TupleElementsNotIn<typename Algorithm::Arguments, OutputArguments>::t>>;
  };

  template<typename OutputArguments, typename Algorithm, typename NextAlgorithm, typename... Algorithms>
  struct OutDependenciesImpl<OutputArguments, std::tuple<Algorithm, NextAlgorithm, Algorithms...>> {
    using previous_t = typename OutDependenciesImpl<OutputArguments, std::tuple<NextAlgorithm, Algorithms...>>::t;
    using t = typename TupleAppend<
      previous_t,
      ScheduledDependencies<
        NextAlgorithm,
        typename TupleElementsNotIn<
          typename ArgumentsNotIn<typename Algorithm::Arguments, std::tuple<NextAlgorithm, Algorithms...>>::t,
          OutputArguments>::t>>::t;
  };

  // Helper to calculate OUT dependencies
  template<typename ConfiguredSequence, typename OutputArguments>
  struct OutDependencies;

  template<typename FirstAlgorithmInSequence, typename... RestOfSequence, typename OutputArguments>
  struct OutDependencies<std::tuple<FirstAlgorithmInSequence, RestOfSequence...>, OutputArguments> {
    using t = typename TupleReverse<typename TupleAppend<
      typename OutDependenciesImpl<
        OutputArguments,
        typename TupleAppend<std::tuple<FirstAlgorithmInSequence, RestOfSequence...>, last_t>::t>::t,
      ScheduledDependencies<FirstAlgorithmInSequence, std::tuple<>>>::t>::t;
  };

  // Consume the algorithms and put their dependencies one by one
  template<typename Algorithms>
  struct InDependenciesImpl;

  template<>
  struct InDependenciesImpl<std::tuple<>> {
    using t = std::tuple<>;
  };

  template<typename Algorithm, typename... Algorithms>
  struct InDependenciesImpl<std::tuple<Algorithm, Algorithms...>> {
    using previous_t = typename InDependenciesImpl<std::tuple<Algorithms...>>::t;
    using t = typename TupleAppend<
      previous_t,
      ScheduledDependencies<
        Algorithm,
        typename ArgumentsNotIn<typename Algorithm::Arguments, std::tuple<Algorithms...>>::t>>::t;
  };

  template<typename ConfiguredSequence>
  using InDependencies = InDependenciesImpl<typename TupleReverse<ConfiguredSequence>::t>;

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
    using previous_t =
      typename ArgumentsTuple<std::tuple<ScheduledDependencies<Algorithm, std::tuple<Args...>>, Algorithms...>>::t;
    using t = typename TupleAppendFirst<Arg, previous_t>::t;
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
    static constexpr void print()
    {
      info_cout << Argument::name << ", ";
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
  struct PrintAlgorithmDependencies<
    std::tuple<ScheduledDependencies<Algorithm, std::tuple<Arguments...>>, Dependencies...>> {
    static constexpr void print()
    {
      info_cout << "Algorithm " << Algorithm::name << ":" << std::endl
                << std::tuple_size<std::tuple<Arguments...>>::value << " dependencies" << std::endl;

      PrintArguments<std::tuple<Arguments...>>::print();
      info_cout << std::endl << std::endl;

      PrintAlgorithmDependencies<std::tuple<Dependencies...>>::print();
    }
  };

  // Print the configured sequence
  template<typename Dependencies>
  struct PrintAlgorithmSequence;

  template<>
  struct PrintAlgorithmSequence<std::tuple<>> {
    static constexpr void print() {};
  };

  template<typename Algorithm, typename... Algorithms>
  struct PrintAlgorithmSequence<std::tuple<Algorithm, Algorithms...>> {
    static constexpr void print()
    {
      info_cout << " " << Algorithm::name << std::endl;
      PrintAlgorithmSequence<std::tuple<Algorithms...>>::print();
    }
  };

  /**
   * @brief Produces a single argument reference.
   */
  template<typename ArgumentsTuple, typename Argument>
  struct ProduceSingleArgument {
    constexpr static Argument& produce(ArgumentsTuple& arguments_tuple)
    {
      Argument& argument = std::get<Argument>(arguments_tuple);
      return argument;
    }
  };

  /**
   * @brief Produces a list of argument references.
   */
  template<typename ArgumentsTuple, typename ArgumentRefManager, typename Arguments>
  struct ProduceArgumentsTupleHelper;

  template<typename ArgumentsTuple, typename ArgumentRefManager, typename... Arguments>
  struct ProduceArgumentsTupleHelper<ArgumentsTuple, ArgumentRefManager, std::tuple<Arguments...>> {
    constexpr static ArgumentRefManager produce(ArgumentsTuple& arguments_tuple)
    {
      return ArgumentRefManager {
        std::forward_as_tuple(ProduceSingleArgument<ArgumentsTuple, Arguments>::produce(arguments_tuple)...)};
    }
  };

  /**
   * @brief Produces a single algorithm with references to arguments.
   */
  template<typename ArgumentsTuple, typename Algorithm>
  struct ProduceArgumentsTuple {
    constexpr static typename Algorithm::arguments_t produce(ArgumentsTuple& arguments_tuple)
    {
      return ProduceArgumentsTupleHelper<
        ArgumentsTuple,
        typename Algorithm::arguments_t,
        typename Algorithm::Arguments>::produce(arguments_tuple);
    }
  };

  /**
   * @brief Runs the sequence tuple (implementation).
   */
  template<
    typename Scheduler,
    typename Functor,
    typename Tuple,
    typename SetSizeArguments,
    typename VisitArguments,
    typename Indices>
  struct RunSequenceTupleImpl;

  template<
    typename Scheduler,
    typename Functor,
    typename Tuple,
    typename... SetSizeArguments,
    typename... VisitArguments>
  struct RunSequenceTupleImpl<
    Scheduler,
    Functor,
    Tuple,
    std::tuple<SetSizeArguments...>,
    std::tuple<VisitArguments...>,
    std::index_sequence<>> {
    constexpr static void run(
      Scheduler& scheduler,
      Functor& functor,
      Tuple& tuple,
      SetSizeArguments&&... set_size_arguments,
      VisitArguments&&... visit_arguments)
    {}
  };

  template<
    typename Scheduler,
    typename Functor,
    typename Tuple,
    typename... SetSizeArguments,
    typename... VisitArguments,
    unsigned long I,
    unsigned long... Is>
  struct RunSequenceTupleImpl<
    Scheduler,
    Functor,
    Tuple,
    std::tuple<SetSizeArguments...>,
    std::tuple<VisitArguments...>,
    std::index_sequence<I, Is...>> {
    constexpr static void run(
      Scheduler& scheduler,
      Functor& functor,
      Tuple& tuple,
      SetSizeArguments&&... set_size_arguments,
      VisitArguments&&... visit_arguments)
    {
      using t = typename std::tuple_element<I, Tuple>::type;

      // Sets the arguments sizes, setups the scheduler and visits the algorithm.
      functor.template set_arguments_size<t>(
        ProduceArgumentsTuple<typename Scheduler::arguments_tuple_t, t>::produce(
          scheduler.argument_manager.arguments_tuple),
        std::forward<SetSizeArguments>(set_size_arguments)...);

      scheduler.template setup<I, t>();

      functor.template visit<t>(
        std::get<I>(tuple),
        ProduceArgumentsTuple<typename Scheduler::arguments_tuple_t, t>::produce(
          scheduler.argument_manager.arguments_tuple),
        std::forward<VisitArguments>(visit_arguments)...);

      RunSequenceTupleImpl<
        Scheduler,
        Functor,
        Tuple,
        std::tuple<SetSizeArguments...>,
        std::tuple<VisitArguments...>,
        std::index_sequence<Is...>>::
        run(
          scheduler,
          functor,
          tuple,
          std::forward<SetSizeArguments>(set_size_arguments)...,
          std::forward<VisitArguments>(visit_arguments)...);
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

  template<
    typename Scheduler,
    typename Functor,
    typename Tuple,
    typename... SetSizeArguments,
    typename... VisitArguments>
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
        std::make_index_sequence<std::tuple_size<Tuple>::value>>::
        run(
          scheduler,
          functor,
          tuple,
          std::forward<SetSizeArguments>(set_size_arguments)...,
          std::forward<VisitArguments>(visit_arguments)...);
    }
  };

  /**
   * @brief Runs the PrChecker for all configured algorithms in the sequence.
   */
  template<typename Functor, typename ConfiguredSequence, typename Arguments>
  struct RunChecker;

  template<typename Functor, typename... Arguments>
  struct RunChecker<Functor, std::tuple<>, std::tuple<Arguments...>> {
    constexpr static void check(const Functor& functor, Arguments&&... arguments) {}
  };

  template<typename Functor, typename Algorithm, typename... Algorithms, typename... Arguments>
  struct RunChecker<Functor, std::tuple<Algorithm, Algorithms...>, std::tuple<Arguments...>> {
    constexpr static void check(const Functor& functor, Arguments&&... arguments)
    {
      functor.template check<Algorithm>(std::forward<Arguments>(arguments)...);

      RunChecker<Functor, std::tuple<Algorithms...>, std::tuple<Arguments...>>::check(
        functor, std::forward<Arguments>(arguments)...);
    }
  };

} // namespace Sch
