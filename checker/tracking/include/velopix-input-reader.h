/** @file velopix-input-reader.h
 *
 * @brief a reader of velopix inputs
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-18
 *
 * 2018-07 Dorothea vom Bruch: updated to run over different track types, 
 * take input from Renato Quagliani's TrackerDumper
 */

#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <tuple>
#include "MCParticle.h"
#include "Common.h"
#include "Logger.h"
#include "InputTools.h"
#include "TrackChecker.h"
#include "MCParticle.h"

class VelopixEvent {
private:
    template<class T>
    static std::string strVector(const T v, const uint vSize, const uint numberOfElements = 5) {
        std::string s = "";
        auto n = std::min(vSize, numberOfElements);
        for (size_t i=0; i<n; ++i) {
            s += std::to_string(v[i]);
            if (i != n-1) s += ", ";
            else if (i == vSize-1) s += "";
            else s += "...";
        }
        return s;
    }

public:
    uint32_t size;
    MCParticles mcps;

    // Constructor
    VelopixEvent() {};
    VelopixEvent(const std::vector<char>& _event, const std::string& trackType, const bool checkFile = true);

    void print() const;

    MCParticles mcparticles() const;
};

std::tuple<bool, std::vector<VelopixEvent>> read_mc_folder(
  const std::string& foldername,
  const std::string& trackType,
  uint number_of_files,
  const uint start_event_offset,
  const bool checkEvents = false
);
 
template<typename t_checker>
void call_pr_checker_impl(
  const std::vector< trackChecker::Tracks >& all_tracks,
  const std::string& folder_name_MC,
  const uint start_event_offset,
  const std::string& trackType
) {
   /* MC information */
  int n_events = all_tracks.size();
  const auto mc_folder_contents = read_mc_folder(folder_name_MC, trackType, n_events, start_event_offset, true );

  if (std::get<0>(mc_folder_contents)) {
    const std::vector<VelopixEvent>& events = std::get<1>(mc_folder_contents);
    t_checker trackChecker {};
    uint64_t evnum = 0; 

    for (const auto& ev: events) {
      const auto& mcps = ev.mcparticles();
      const std::vector<MCParticle>& mcps_vector = ev.mcps;
      MCAssociator mcassoc(mcps);

      trackChecker(all_tracks[evnum], mcassoc, mcps);

      // Check all tracks for duplicate LHCb IDs
      uint i_track = 0;
      for (auto& track : all_tracks[evnum]) {
        auto ids = track.ids();
        std::sort(std::begin(ids), std::end(ids));
        bool containsDuplicates = (std::unique(std::begin(ids), std::end(ids))) != std::end(ids);

        if (containsDuplicates) {
          warning_cout << "WARNING: Track #" << std::dec << i_track << " contains duplicate LHCb IDs" << std::endl;
          for (auto id : ids) {
            warning_cout << std::hex << "0x" << id << ", ";
          }
          warning_cout << std::endl << std::endl << std::hex;
        }
        i_track++;
      }
      
      ++evnum;
    }
  }
}
