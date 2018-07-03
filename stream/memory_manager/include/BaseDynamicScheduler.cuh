#pragma once

#include "MemoryManager.cuh"

struct BaseDynamicScheduler {
  MemoryManager memory_manager;
  std::vector<std::string> argument_names;
  std::vector<std::vector<uint>> sequence_arguments;
  int current_sequence_step = 0;
  // Type that determines the tags to free and initialize on each step
  std::vector<std::vector<uint>> tags_to_initialize;
  std::vector<std::vector<uint>> tags_to_free;

  BaseDynamicScheduler() = default;

  BaseDynamicScheduler(const std::vector<std::string>& param_argument_names,
    const std::vector<std::vector<uint>>& param_sequence_arguments,
    const size_t reserved_mb)
    : argument_names(param_argument_names),
    sequence_arguments(param_sequence_arguments) {
    // Generate the helper arguments vector
    generate_tags();
    // Set max mb to memory_manager
    memory_manager.set_reserved_memory(reserved_mb);
  }

  /**
   * @brief Helper function to generate vector of MemoryArgument
   *        from argument_sizes and sequence_arguments.
   */
  void generate_tags() {
    tags_to_initialize.clear();
    tags_to_free.clear();
    std::vector<uint> tags_initialized;
    std::vector<uint> tags_freed;

    // Iterate over sequence and populate first appeareance of algorithms
    for (int i=0; i<sequence_arguments.size(); ++i) {
      std::vector<uint> initialize;
      for (uint argument_number : sequence_arguments[i]) {
        if (std::find(std::begin(tags_initialized), std::end(tags_initialized), argument_number)
          == std::end(tags_initialized)) {
          initialize.push_back(argument_number);
          tags_initialized.push_back(argument_number);
        }
      }
      tags_to_initialize.push_back(initialize);
    }

    // Iterate over sequence in reverse and populate last appeareance of algorithms
    for (auto it=sequence_arguments.rbegin(); it!=sequence_arguments.rend(); ++it) {
      std::vector<uint> freed;
      for (uint argument_number : *it) {
        if (std::find(std::begin(tags_freed), std::end(tags_freed), argument_number)
          == std::end(tags_freed)) {
          freed.push_back(argument_number);
          tags_freed.push_back(argument_number);
        }
      }
      tags_to_free.push_back(freed);
    }
    std::reverse(std::begin(tags_to_free), std::end(tags_to_free));

    // Sanity check
    if (tags_to_initialize.size() != sequence_arguments.size() ||
      tags_to_free.size() != sequence_arguments.size()) {
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
  template<unsigned long N>
  void setup_next(
    const std::array<size_t, N>& argument_sizes,
    std::array<uint, N>& argument_offsets,
    const int check_sequence_step = -1,
    const bool do_print = false
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
    for (uint tag : tags_to_initialize[current_sequence_step]) {
      argument_offsets[tag] = memory_manager.reserve((int) tag, argument_sizes[tag]);
    }

    // Print memory manager state
    if (do_print) {
      memory_manager.print(argument_names, current_sequence_step);
    }

    // Move to next step
    ++current_sequence_step;
  }
};
