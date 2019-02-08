#ifndef MDFREADER_H
#define MDFREADER_H 1

#include <map>
#include <string>

#include "InputReader.h"

struct MDFReader : public EventReader {

  MDFReader(FolderMap folders) : EventReader(std::move(folders)) {}

  /**
   * @brief Reads files from the specified folder, starting from an event offset.
   */
  void read_events(uint number_of_events_requested = 0, uint start_event_offset = 0) override;
};
#endif
