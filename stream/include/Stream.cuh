#pragma once

#include <iostream>
#include <vector>
#include "../../main/include/Common.h"
#include "../../main/include/Logger.h"
#include "CalculatePhiAndSort.cuh"
#include "SearchByTriplet.cuh"
#include "CalculateVeloStates.cuh"
#include "Helper.cuh"

struct Stream {
  cudaStream_t stream;
  bool do_print_timing;

  Stream(bool do_print_timing = true) :
    do_print_timing(do_print_timing) {
    cudaStreamCreate(&stream);
  }

  ~Stream() {
    cudaStreamDestroy(stream);
  }

  cudaError_t operator()(
    const std::vector<std::vector<uint8_t>>& input,
    unsigned int start_event,
    unsigned int number_of_events
  );

  void print_timing(
    const unsigned int number_of_events,
    const std::vector<float>& times
  );
};
