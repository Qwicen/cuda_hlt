#include "StreamWrapper.cuh"
#include "Stream.cuh"

void StreamWrapper::initialize_streams(
  const uint n,
  const uint number_of_events,
  const bool do_check,
  const bool do_simplified_kalman_filter,
  const bool print_memory_usage,
  const bool run_on_x86,
  const std::string& folder_name_MC,
  const uint start_event_offset,
  const size_t reserve_mb,
  const Constants& constants
) {
  for (uint i=0; i<n; ++i) {
    streams.push_back(new Stream());
  }

  for (int i=0; i<streams.size(); ++i) {
    streams[i]->initialize(
      number_of_events,
      do_check,
      do_simplified_kalman_filter,
      print_memory_usage,
      run_on_x86,
      folder_name_MC,
      start_event_offset,
      reserve_mb,
      i,
      constants
    );

    // Memory consumption
    size_t free_byte;
    size_t total_byte;
    cudaCheck(cudaMemGetInfo(&free_byte, &total_byte));
    float free_percent = (float)free_byte / total_byte * 100;
    float used_percent = (float)(total_byte - free_byte) / total_byte * 100;
    verbose_cout << "GPU memory: " << free_percent << " percent free, "
      << used_percent << " percent used " << std::endl;
  }
}

void StreamWrapper::run_stream(const uint i, const RuntimeOptions& runtime_options) {
  auto& s = *(streams[i]);
  s.run_sequence(runtime_options);
}

void StreamWrapper::run_monte_carlo_test(const uint i, const uint number_of_events_requested) {
  auto& s = *(streams[i]);
  s.run_monte_carlo_test(number_of_events_requested);
}

StreamWrapper::~StreamWrapper() {
  for (auto& stream : streams) {
    delete stream;
  }
}
