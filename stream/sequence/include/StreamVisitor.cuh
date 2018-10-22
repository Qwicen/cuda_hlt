#pragma once

#include "RuntimeOptions.cuh"
#include "Constants.cuh"
#include "ArgumentManager.cuh"
#include "HostBuffers.cuh"

struct StreamVisitor {
  template<typename T>
  void visit(
    T& state,
    const int sequence_step,
    const RuntimeOptions& runtime_options,
    const Constants& constants,
    ArgumentManager<argument_tuple_t>& arguments,
    HostBuffers& host_buffers,
    cudaStream_t& cuda_stream,
    cudaEvent_t& cuda_generic_event);
};
