#include "InputReader.h"

Reader::Reader(const std::string& folder_name) : folder_name(folder_name) {
  if (!exists_test(folder_name)) {
    throw StrException("Folder " + folder_name + " does not exist.");
  }
}

std::vector<char> GeometryReader::read_geometry(const std::string& filename) {
  std::vector<char> geometry;
  ::read_geometry(folder_name + "/" + filename, geometry);
  return geometry;
}

std::vector<char> UTMagnetToolReader::read_UT_magnet_tool() {
  std::vector<char> ut_magnet_tool;
  ::read_UT_magnet_tool(folder_name, ut_magnet_tool);
  return ut_magnet_tool;
}

void EventReader::read_events(uint number_of_events_requested, uint start_event_offset) {
  std::vector<char> events;
  std::vector<uint> event_offsets;

  read_folder(
    folder_name,
    number_of_events_requested,
    events,
    event_offsets,
    start_event_offset
  );

  check_events(events, event_offsets, number_of_events_requested);

  // TODO Remove: Temporal check to understand if number_of_events_requested is the same as number_of_events
  const int number_of_events = event_offsets.size() - 1;
  if (number_of_events_requested != number_of_events) {
    throw StrException("Number of events requested differs from number of events read.");
  }

  // Copy raw data to pinned host memory
  cudaCheck(cudaMallocHost((void**)&host_events, events.size()));
  cudaCheck(cudaMallocHost((void**)&host_event_offsets, event_offsets.size() * sizeof(uint)));
  std::copy_n(std::begin(events), events.size(), host_events);
  std::copy_n(std::begin(event_offsets), event_offsets.size(), host_event_offsets);

  host_events_size = events.size();
  host_event_offsets_size = event_offsets.size();
}

