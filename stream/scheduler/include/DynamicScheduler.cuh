#pragma once

#include "MemoryManager.cuh"
#include "Argument.cuh"

template<typename T, typename R>
struct DynamicScheduler {
  MemoryManager memory_manager;
  std::array<std::string, std::tuple_size<T>::value> sequence_names;
  std::array<std::string, std::tuple_size<R>::value> argument_names;
  std::vector<std::vector<int>> sequence_dependencies;
  std::vector<int> sequence_output_arguments;
  int current_sequence_step = 0;
  bool do_print = false;
  // Type that determines the tags to free and initialize on each step
  std::vector<std::vector<int>> tags_to_initialize;
  std::vector<std::vector<int>> tags_to_free;

  DynamicScheduler() = default;

  DynamicScheduler(
    // const std::array<std::string, std::tuple_size<T>::value>& param_sequence_names,
    const std::array<std::string, std::tuple_size<R>::value>& param_argument_names,
    const std::vector<std::vector<int>>& param_sequence_dependencies,
    const std::vector<int> param_sequence_output_arguments,
    const size_t reserved_mb,
    const bool param_do_print)
  :
    // sequence_names(param_sequence_names),
    argument_names(param_argument_names),
    sequence_dependencies(param_sequence_dependencies),
    sequence_output_arguments(param_sequence_output_arguments),
    do_print(param_do_print) {
    // Generate the helper arguments vector
    generate_tags();
    // Set max mb to memory_manager
    memory_manager.set_reserved_memory(reserved_mb);
  }

  /**
   * @brief Helper function to generate vector of MemoryArgument
   *        from argument_sizes and sequence_dependencies.
   */
  void generate_tags() {
    tags_to_initialize.clear();
    tags_to_free.clear();
    std::vector<int> tags_initialized;
    std::vector<int> tags_freed;

    // Iterate over sequence and populate first appeareance of algorithms
    for (int i=0; i<sequence_dependencies.size(); ++i) {
      std::vector<int> initialize;
      for (int argument_number : sequence_dependencies[i]) {
        if (std::find(std::begin(tags_initialized), std::end(tags_initialized), argument_number)
          == std::end(tags_initialized)) {
          initialize.push_back(argument_number);
          tags_initialized.push_back(argument_number);
        }
      }
      tags_to_initialize.push_back(initialize);
    }

    // Iterate over sequence in reverse and populate last appeareance of algorithms
    for (auto it=sequence_dependencies.rbegin(); it!=sequence_dependencies.rend(); ++it) {
      std::vector<int> freed;
      for (int argument_number : *it) {
        if (std::find(std::begin(tags_freed), std::end(tags_freed), argument_number)
          == std::end(tags_freed)) {
          // Never free arguments in sequence_output_arguments
          if (std::find(std::begin(sequence_output_arguments), std::end(sequence_output_arguments),
            argument_number) == std::end(sequence_output_arguments)) {
            freed.push_back(argument_number);
            tags_freed.push_back(argument_number);
          }
        }
      }
      tags_to_free.push_back(freed);
    }
    std::reverse(std::begin(tags_to_free), std::end(tags_to_free));

    // Sanity check
    if (tags_to_initialize.size() != sequence_dependencies.size() ||
      tags_to_free.size() != sequence_dependencies.size()) {
      throw StrException("Generate arguments: tags to initialize, tags to free size mismatch.");
    }
  }

  void reset() {
    current_sequence_step = 0;
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
  void setup_next(
    ArgumentManager<R>& arguments,
    const int check_sequence_step = -1
  ) {
    assert(check_sequence_step==-1 || (current_sequence_step == check_sequence_step));

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
