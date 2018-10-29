#pragma once

#include "MemoryManager.cuh"
#include "SchedulerMachinery.cuh"

template<typename ConfiguredSequence, typename AlgorithmsDependencies, typename OutputArguments>
struct Scheduler {
  // Dependencies calculated at compile time
  // Determines what to free (out_deps) and malloc (in_deps)
  // at every iteration.
  using in_deps_t = typename Sch::in_dependencies<ConfiguredSequence, AlgorithmsDependencies>::t;
  using out_deps_t = typename Sch::out_dependencies<ConfiguredSequence, OutputArguments, AlgorithmsDependencies>::t;

  in_deps_t in_deps;
  out_deps_t out_deps;
  bool do_print = false;
  MemoryManager memory_manager;

  Scheduler() = default;

  Scheduler(const bool param_do_print, const size_t reserved_mb) : do_print(param_do_print) {
    // Set max mb to memory_manager
    memory_manager.set_reserved_memory(reserved_mb);
  }

  void reset() {
    memory_manager.free_all();
  }

  /**
   * @brief Runs a step of the scheduler and determines
   *        the offset for each argument. The size
   *        of the offset vector must be the same as the number
   *        of input arguments for that step of the sequence.
   *        
   *        The parameter check_sequence_step can be used to assert
   *        the sequence being setup is the one intended.
   *        
   *        This function should always be invoked, even when it is
   *        known there are no tags to reserve or free on this step.
   *        It performs a check on the current sequence item and 
   *        increments the sequence step.
   */
  template<unsigned long I, typename T>
  void setup {
    // in_deps and out_deps should be in order
    // and index I should contain algorithm type T
    using in_algorithm = std::get<I>(in_deps)::Algorithm;
    using out_algorithm = std::get<I>(out_deps)::Algorithm;

    static_assert(std::is_same<T, in_algorithm>::value, "Scheduler index mismatch (in_algorithm)");
    static_assert(std::is_same<T, out_algorithm>::value, "Scheduler index mismatch (out_algorithm)");

    // Free all tags in previous step
    if (current_sequence_step != 0) {
      for (auto tag : tags_to_free[current_sequence_step-1]) {
        memory_manager.free(tag);
      }
    }

    // Reserve space for all tags
    // that need to be initialized on this step
    for (auto tag : tags_to_initialize[current_sequence_step]) {
      const auto requested_size = arguments.size(tag);
      arguments.set_offset(tag, memory_manager.reserve(tag, requested_size));
    }

    // Print memory manager state
    if (do_print) {
      memory_manager.print<T, R>(sequence_names, argument_names, current_sequence_step);
    }

    // Move to next step
    ++current_sequence_step;
  }
};
