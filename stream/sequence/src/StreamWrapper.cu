#include "StreamWrapper.cuh"
#include "Stream.cuh"

void StreamWrapper::initialize_streams(
  const uint n,
  const std::vector<char>& velopix_geometry,
  const std::vector<char>& ut_boards,
  const std::vector<char>& ut_geometry,
  const std::vector<char>& ut_magnet_tool,
  const std::vector<char>& scifi_geometry,
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
      velopix_geometry,
      ut_boards,
      ut_geometry,
      ut_magnet_tool,
      scifi_geometry,
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

void StreamWrapper::run_stream(
  const uint i,
  char* host_velopix_events,
  uint* host_velopix_event_offsets,
  const size_t velopix_events_size,
  const size_t velopix_event_offsets_size,
  char* host_ut_events,
  uint* host_ut_event_offsets,
  const size_t ut_events_size,
  const size_t ut_event_offsets_size,
  char* host_scifi_events,
  uint* host_scifi_event_offsets,
  const size_t scifi_events_size,
  const size_t scifi_event_offsets_size,
  const uint number_of_events,
  const uint number_of_repetitions
) {
  auto& s = *(streams[i]);
  s.run_sequence(
    i,
    host_velopix_events,
    host_velopix_event_offsets,
    velopix_events_size,
    velopix_event_offsets_size,
    host_ut_events,
    host_ut_event_offsets,
    ut_events_size,
    ut_event_offsets_size,
    host_scifi_events,
    host_scifi_event_offsets,
    scifi_events_size,
    scifi_event_offsets_size,
    number_of_events,
    number_of_repetitions
  );
}

StreamWrapper::~StreamWrapper() {
  for (auto& stream : streams) {
    delete stream;
  }
}
