#include "../include/KernelInvoker.cuh"
#include "../../stream/include/Stream.cuh"

cudaError_t invokeParallelSearch(
  const std::vector<std::vector<uint8_t>>& input,
  std::vector<std::vector<uint8_t>>& output
) {
  Stream s;

  s(input, 0, input.size());

  return cudaSuccess;
}
