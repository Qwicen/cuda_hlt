#pragma once

#include "../../memory_manager/include/MemoryManager.cuh"

struct BaseStaticScheduler {
  struct MemoryArgument {
    // Argument number
    int tag;
    size_t size;
    // First and last algorithm appeareance
    int first_algorithm;
    int last_algorithm;

    MemoryArgument() = default;
    MemoryArgument(int param_tag, size_t param_size,
      int param_first_algorithm=-1, int param_last_algorithm=-1)
      : tag(param_tag), size(param_size),
      first_algorithm(param_first_algorithm),
      last_algorithm(param_last_algorithm) {}
  };
  std::vector<MemoryArgument> arguments;

  MemoryManager* memory_manager;
  std::vector<size_t> argument_sizes;
  std::vector<std::string> argument_names;
  std::vector<std::vector<uint>> sequence_arguments;

  BaseStaticScheduler() = default;

  BaseStaticScheduler(std::vector<size_t> param_argument_sizes,
    std::vector<std::string> param_argument_names,
    std::vector<std::vector<uint>> param_sequence_arguments)
    : argument_sizes(param_argument_sizes),
    argument_names(param_argument_names),
    sequence_arguments(param_sequence_arguments) {
    // By default, use base MemoryManager
    memory_manager = new MemoryManager();
    // Generate the helper arguments vector
    generate_arguments_vector();
  }

  ~BaseStaticScheduler() {
    if (memory_manager != nullptr) {
      delete memory_manager;
    }
  }

  /**
   * @brief Helper function to generate vector of MemoryArgument
   *        from argument_sizes and sequence_arguments.
   */
  void generate_arguments_vector() {
    for (int i=0; i<argument_sizes.size(); ++i) {
      arguments.push_back(MemoryArgument{i, argument_sizes[i]});
    }

    // Iterate over sequence and populate first / last algorithms
    for (int i=0; i<sequence_arguments.size(); ++i) {
      for (auto& argument_number : sequence_arguments[i]) {
        auto& argument = arguments[argument_number];
        if (argument.first_algorithm == -1) {
          argument.first_algorithm = i;
        }
        argument.last_algorithm = i;
      }
    }

    // Sanity check
    for (auto& argument : arguments) {
      if (argument.first_algorithm == -1 ||
        argument.last_algorithm == -1) {
        throw StrException("Generate arguments: At least one argument has " +
          std::string("first / last algorithm not populated (argument ") +
          argument_names[argument.tag] + ")");
      }
    }
  }

  /**
   * @brief Changes the memory manager.
   */
  template<typename T>
  void set_memory_manager() {
    if (memory_manager != nullptr) {
      delete memory_manager;
    }
    memory_manager = new T();
  }

  /**
   * @brief Runs a static scheduler and determines the memory
   *        required and the offset for each argument. The size
   *        of the offset vector must be the same as argument_sizes.
   */
  virtual std::tuple<size_t, std::vector<uint>> solve() {
    // Return type
    std::vector<uint> offsets (argument_sizes.size(), 0);

    const auto sequence_length = sequence_arguments.size();
    for (uint i=0; i<sequence_length; ++i) {
      // Reserve memory for types created on this algorithm
      for (auto& argument : arguments) {
        if (argument.first_algorithm == i) {
          offsets[argument.tag] = memory_manager->reserve(argument.tag, argument.size);
        }
      }

      memory_manager->print(argument_names, i);

      // Free memory for types not used anymore after this algorithm
      for (auto& argument : arguments) {
        if (argument.last_algorithm == i) {
          memory_manager->free(argument.tag);
        }
      }
    }

    return {memory_manager->total_memory_required, offsets};
  }
};
