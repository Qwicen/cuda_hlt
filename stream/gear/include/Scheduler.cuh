#pragma once

#include "MemoryManager.cuh"
#include "SchedulerMachinery.cuh"
#include "ArgumentManager.cuh"

template<typename ConfiguredSequence, typename AlgorithmsDependencies, typename OutputArguments>
struct Scheduler {
  // Dependencies calculated at compile time
  // Determines what to free (out_deps) and reserve (in_deps)
  // at every iteration.
  using in_deps_t = typename Sch::in_dependencies<ConfiguredSequence, AlgorithmsDependencies>::t;
  using out_deps_t = typename Sch::out_dependencies<ConfiguredSequence, OutputArguments, AlgorithmsDependencies>::t;
  using arguments_tuple_t = typename Sch::ArgumentsTuple<in_deps_t>::t;
  using argument_manager_t = ArgumentManager<arguments_tuple_t>;

  in_deps_t in_deps;
  out_deps_t out_deps;
  MemoryManager memory_manager;
  argument_manager_t argument_manager;
  bool do_print = false;

  Scheduler() = default;

  Scheduler(
    const bool param_do_print,
    const size_t reserved_mb,
    char* base_pointer)
  : do_print(param_do_print) {
    // Set max mb to memory_manager
    memory_manager.set_reserved_memory(reserved_mb);
    argument_manager.set_base_pointer(base_pointer);

    // std::cout << "IN deps" << std::endl;
    // Sch::print_algorithm_dependencies<in_deps_t>::print();

    // std::cout << "OUT deps" << std::endl;
    // Sch::print_algorithm_dependencies<out_deps_t>::print();
  }

  /**
   * @brief Returns the argument manager of the scheduler.
   */
  argument_manager_t& arguments() {
    return argument_manager;
  }

  /**
   * @brief Resets the memory manager.
   */
  void reset() {
    memory_manager.free_all();
  }

  /**
   * @brief Runs a step of the scheduler and determines
   *        the offset for each argument.
   *        
   *        The sequence is asserted at compile time to run the
   *        expected iteration and reserve the expected types.
   *        
   *        This function should always be invoked, even when it is
   *        known there are no tags to reserve or free on this step.
   */
  template<unsigned long I, typename T>
  void setup() {
    // in dependencies: Dependencies to be reserved
    // out dependencies: Dependencies to be free'd
    // 
    // in_deps and out_deps should be in order
    // and index I should contain algorithm type T
    using in_deps_I_t = typename std::tuple_element<I, in_deps_t>::type;
    using out_deps_I_t = typename std::tuple_element<I, out_deps_t>::type;
    using in_algorithm = typename in_deps_I_t::Algorithm;
    using in_arguments = typename in_deps_I_t::Arguments;
    using out_algorithm = typename out_deps_I_t::Algorithm;
    using out_arguments = typename out_deps_I_t::Arguments;

    static_assert(std::is_same<T, in_algorithm>::value, "Scheduler index mismatch (in_algorithm)");
    static_assert(std::is_same<T, out_algorithm>::value, "Scheduler index mismatch (out_algorithm)");

    // Free all arguments in out_dependencies    
    MemoryManagerFree<out_arguments>::free(memory_manager);

    // Reserve all arguments in in_dependencies
    MemoryManagerReserve<argument_manager_t, in_arguments>::reserve(memory_manager, argument_manager);

    // Print memory manager state
    if (do_print) {
      info_cout << "Sequence step " << I << " \"" << T::name << "\":" << std::endl;
      memory_manager.print();
    }
  }
};
