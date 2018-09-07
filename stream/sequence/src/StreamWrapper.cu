#include "StreamWrapper.cuh"
#include "Stream.cuh"

void StreamWrapper::initialize_streams(
  const uint n,
  const std::vector<char>& velopix_geometry,
  const PrUTMagnetTool* host_ut_magnet_tool,
  const uint number_of_events,
  const bool transmit_device_to_host,
  const bool do_check,
  const bool do_simplified_kalman_filter,
  const bool print_memory_usage,
  const bool run_on_x86,
  const std::string& folder_name_MC,
  const uint start_event_offset,
  const size_t reserve_mb,
  const GpuConstants& gpu_constants
) {
  for (uint i=0; i<n; ++i) {
    streams.push_back(new Stream());
  }

  for (int i=0; i<streams.size(); ++i) {
    streams[i]->initialize(
      velopix_geometry,
      host_ut_magnet_tool,
      number_of_events,
      transmit_device_to_host,
      do_check,
      do_simplified_kalman_filter,
      print_memory_usage,
      run_on_x86,
      folder_name_MC,
      start_event_offset,
      reserve_mb,
      i,
      gpu_constants
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
  VeloUTTracking::HitsSoA *host_ut_hits_events,
  const PrUTMagnetTool* host_ut_magnet_tool,
  ForwardTracking::HitsSoAFwd *hits_layers_events_ft,
  const uint32_t n_hits_layers_events_ft[][ForwardTracking::n_layers],
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
    host_ut_hits_events,
    host_ut_magnet_tool,
    hits_layers_events_ft,
    n_hits_layers_events_ft,
    number_of_events,
    number_of_repetitions
  );
}

StreamWrapper::~StreamWrapper() {
  for (auto& stream : streams) {
    delete stream;
  }
}
