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
    const std::vector<char>& events,
    const std::vector<unsigned int>& event_offsets,
    const std::vector<unsigned int>& hit_offsets,
    unsigned int start_event,
    unsigned int number_of_events,
    unsigned int number_of_repetitions
  );

  void print_timing(
    const unsigned int number_of_events,
    const std::vector<std::pair<std::string, float>>& times
  );
};
