#pragma once

#include "RuntimeOptions.h"
#include "Logger.h"
#include "Constants.cuh"
#include "Argument.cuh"
#include "HostBuffers.cuh"
#include "Scheduler.cuh"
#include "ConfiguredSequence.cuh"
#include "AlgorithmDependencies.cuh"
#include "Arguments.cuh"

struct SequenceVisitor {
  using scheduler_t = Scheduler<configured_sequence_t, algorithms_dependencies_t, output_arguments_t>;
  using argument_manager_t = ArgumentManager<scheduler_t::arguments_tuple_t>;

  /**
   * @brief Sets the size of the required arguments.
   */
  template<typename T>
  void set_arguments_size(
    const RuntimeOptions& runtime_options,
    const Constants& constants,
    const HostBuffers& host_buffers,
    argument_manager_t& arguments) {}

  /**
   * @brief   Invokes the algorithm specified in the state.
   * @details The specialization may contain more logic, like
   *          reading some data into host_buffers, waiting for a certain
   *          CUDA event, or some debug code.
   */
  template<typename T>
  void visit(
    T& state,
    const RuntimeOptions& runtime_options,
    const Constants& constants,
    argument_manager_t& arguments,
    HostBuffers& host_buffers,
    cudaStream_t& cuda_stream,
    cudaEvent_t& cuda_generic_event);
};
