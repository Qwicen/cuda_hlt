#include "InputTools.h"
#include "Common.h"
#include "Tools.h"
#include "CudaCommon.h"
#include <string>
#include <algorithm>

struct Reader {
  std::string folder_name;

  /**
   * @brief Sets the folder name parameter and check the folder exists.
   */
  Reader(const std::string& folder_name);
};

struct GeometryReader : public Reader {
  GeometryReader(const std::string& folder_name) : Reader(folder_name) {}

  /**
   * @brief Reads a geometry file from the specified folder.
   */
  std::vector<char> read_geometry(const std::string& filename) const;
};

struct UTMagnetToolReader : public Reader {
  UTMagnetToolReader(const std::string& folder_name) : Reader(folder_name) {}

  /**
   * @brief Reads the UT magnet tool from the specified folder.
   */
  std::vector<char> read_UT_magnet_tool() const;
};

struct EventReader : public Reader {
  char* host_events;
  uint* host_event_offsets;
  size_t host_events_size;
  size_t host_event_offsets_size;

  EventReader(const std::string& folder_name) : Reader(folder_name) {}

  /**
   * @brief Reads files from the specified folder, starting from an event offset.
   */
  virtual void read_events(uint number_of_events_requested=0, uint start_event_offset=0);

  /**
   * @brief Checks the consistency of the read buffers.
   */
  virtual bool check_events(
    const std::vector<char>& events,
    const std::vector<uint>& event_offsets,
    uint number_of_events_requested
  ) {
    return true;
  }
};

struct VeloReader : public EventReader {
  VeloReader(const std::string& folder_name) : EventReader(folder_name) {}

  /**
   * @brief Checks the consistency of Velo raw data.
   */
  bool check_events(
    const std::vector<char>& events,
    const std::vector<uint>& event_offsets,
    uint number_of_files
  ) override {
    return check_velopix_events(events, event_offsets, number_of_files);
  }
};

// TODO: Develop an UT event checker
