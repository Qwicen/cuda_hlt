#include "../include/StreamWrapper.cuh"
#include "../include/Stream.cuh"
#include "../../../main/include/Logger.h"
#include "../../../main/include/Common.h"

void StreamWrapper::initialize_streams(
  const uint n,
  const std::vector<char>& velopix_events,
  const std::vector<uint>& velopix_event_offsets,
  const std::vector<char>& geometry,
  const uint number_of_events,
  const bool transmit_host_to_device,
  const bool transmit_device_to_host,
  const bool do_check,
  const bool do_simplified_kalman_filter,
  const bool print_individual_rates,
  const std::string& folder_name_MC
) {
  for (uint i=0; i<n; ++i) {
    streams.push_back(new Stream());
  }

  for (int i=0; i<streams.size(); ++i) {
    streams[i]->initialize(
      velopix_events,
      velopix_event_offsets,
      geometry,
      number_of_events,
      transmit_host_to_device,
      transmit_device_to_host,
      do_check,
      do_simplified_kalman_filter,
      print_individual_rates,
      folder_name_MC,
      i
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

void StreamWrapper::run_stream(
  const uint i,
  char* host_velopix_events_pinned,
  uint* host_velopix_event_offsets_pinned,
  const size_t velopix_events_size,
  const size_t velopix_event_offsets_size,
  const uint number_of_events,
  const uint number_of_repetitions
) {
  auto& s = *(streams[i]);
  s(
    host_velopix_events_pinned,
    host_velopix_event_offsets_pinned,
    velopix_events_size,
    velopix_event_offsets_size,
    number_of_events,
    number_of_repetitions
  );
}

StreamWrapper::~StreamWrapper() {
  for (auto& stream : streams) {
    delete stream;
  }
}
