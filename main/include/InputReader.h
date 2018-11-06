#ifndef INPUTREADER_H
#define INPUTREADER_H 1

#include "InputTools.h"
#include "Common.h"
#include "BankTypes.h"
#include "Tools.h"
#include "CudaCommon.h"
#include <string>
#include <algorithm>
#include <unordered_set>
#include <gsl-lite.hpp>

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
  std::vector<char> read_geometry(const std::string& filename);
};

struct UTMagnetToolReader : public Reader {
  UTMagnetToolReader(const std::string& folder_name) : Reader(folder_name) {}

  /**
   * @brief Reads the UT magnet tool from the specified folder.
   */
  std::vector<char> read_UT_magnet_tool();
};

using FolderMap = std::map<BankTypes, std::string>;

struct EventReader : public Reader {
  EventReader(FolderMap folders)
     : Reader(begin(folders)->second),
       m_folders{std::move(folders)} {}

  gsl::span<char> events(BankTypes type) {
     auto it = m_events.find(type);
     if (it == end(m_events)) {
        return {};
     } else {
        return it->second.first;
     }
  }

  gsl::span<uint> offsets(BankTypes type) {
     auto it = m_events.find(type);
     if (it == end(m_events)) {
        return {};
     } else {
        return it->second.second;
     }
  }

  /**
   * @brief Reads files from the specified folder, starting from an event offset.
   */
  virtual void read_events(uint number_of_events_requested=0, uint start_event_offset=0);

  /**
   * @brief Checks the consistency of the read buffers.
   */
  virtual bool check_events(
    BankTypes type,
    const std::vector<char>& events,
    const std::vector<uint>& event_offsets,
    uint number_of_events_requested
  ) const;

protected:

   std::string folder(BankTypes type) const {
     auto it = m_folders.find(type);
     if (it == end(m_folders)) {
       return {};
     } else {
       return it->second;
     }
   }

   std::unordered_set<BankTypes> types() const {
      std::unordered_set<BankTypes> r;
      for (const auto& entry : m_folders) {
         r.emplace(entry.first);
      }
      return r;
   }

   bool add_events(BankTypes type, gsl::span<char> events, gsl::span<uint> offsets) {
      auto r = m_events.emplace(type, std::make_pair(std::move(events), std::move(offsets)));
      return r.second;
   }

private:
   std::map<BankTypes, std::pair<gsl::span<char>, gsl::span<uint>>> m_events;
   std::map<BankTypes, std::string> m_folders;
};

#endif
