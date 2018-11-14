#include "StreamWrapper.cuh"
#include "Stream.cuh"

void StreamWrapper::initialize_streams(
  const uint n,
  const uint number_of_events,
  const bool print_memory_usage,
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
      print_memory_usage,
      start_event_offset,
      reserve_mb,
      i,
      constants
    );
  }
}

void StreamWrapper::run_stream(const uint i, const RuntimeOptions& runtime_options) {
  auto& s = *(streams[i]);
  s.run_sequence(runtime_options);
}

void StreamWrapper::run_monte_carlo_test(
  const uint i,
  const std::string& mc_folder,
  const std::string& mc_pv_folder,
  const uint number_of_events_requested)
{
  auto& s = *(streams[i]);
  s.run_monte_carlo_test(mc_folder, mc_pv_folder, number_of_events_requested);
}

StreamWrapper::~StreamWrapper() {
  for (auto& stream : streams) {
    delete stream;
  }
}
