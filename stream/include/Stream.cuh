#pragma once

#include <iostream>
#include <vector>
#include "../../main/include/Common.h"
#include "../../main/include/Logger.h"
#include "../../main/include/Timer.h"

class Timer;

struct Stream {
  cudaStream_t stream;
  unsigned int stream_number;
  bool do_print_timing;

  Stream(
    unsigned int stream_number = 0,
    bool do_print_timing = true
  ) :
    stream_number(stream_number),
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
