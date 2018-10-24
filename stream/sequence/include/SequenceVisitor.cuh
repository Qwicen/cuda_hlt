#pragma once

#include "RuntimeOptions.h"
#include "Logger.h"
#include "Constants.cuh"
#include "Argument.cuh"
#include "HostBuffers.cuh"
#include "SequenceSetup.cuh"
#include "DynamicScheduler.cuh"

struct SequenceVisitor {
  template<typename T>
  void visit(
    T& state,
    const int sequence_step,
    const RuntimeOptions& runtime_options,
    const Constants& constants,
    ArgumentManager<argument_tuple_t>& arguments,
    DynamicScheduler<sequence_t, argument_tuple_t>& scheduler,
    HostBuffers& host_buffers,
    cudaStream_t& cuda_stream,
    cudaEvent_t& cuda_generic_event);
};
