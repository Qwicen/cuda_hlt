#pragma once

#include <string>
#include <vector>
#include "TrackChecker.h"
#include "MCEvent.h"
#include "InputTools.h"

struct CheckerInvoker {
  std::string mc_folder;
  uint start_event_offset;
  uint number_of_requested_events;
  uint number_of_selected_events;
  bool check_events;
  bool is_mc_folder_populated;
  MCEvents mc_events;
  MCEvents selected_mc_events;

  CheckerInvoker (
    const std::string& param_mc_folder,
    const uint param_start_event_offset,
    const uint* event_list,
    const uint param_number_of_requested_events,
    const uint param_number_of_selected_events,
    const bool param_check_events = false) :
    mc_folder(param_mc_folder),
    start_event_offset(param_start_event_offset),
    number_of_requested_events(param_number_of_requested_events),
    number_of_selected_events(param_number_of_selected_events),
      check_events(param_check_events)
  {
    const auto folder_contents = read_mc_folder();

    is_mc_folder_populated = std::get<0>(folder_contents);
    mc_events = std::get<1>(folder_contents);

    // events selected by global event cuts
    for ( int i = 0; i < number_of_selected_events; i++ ) {
      const uint event = event_list[i];
      MCEvent mc_event = mc_events[event];
      selected_mc_events.push_back(mc_event);
    }
  }

  std::tuple<bool, MCEvents> read_mc_folder() const;

  template<typename T>
  void check(
    const uint start_event_offset,
    const std::vector<trackChecker::Tracks>& tracks) const
  {
    if (is_mc_folder_populated) {
      T trackChecker {};
#ifdef WITH_ROOT
      trackChecker.histos.initHistos(trackChecker.histo_categories() );
#endif

      for (int evnum = 0; evnum < selected_mc_events.size(); ++evnum) {
        const auto& mc_event = selected_mc_events[evnum];
        const auto& event_tracks = tracks[evnum];

        const auto& mcps = mc_event.mc_particles<T>();
        MCAssociator mcassoc {mcps};

        trackChecker(event_tracks, mcassoc, mcps);

        // Check all tracks for duplicate LHCb IDs
        for (int i_track = 0; i_track < event_tracks.size(); ++i_track) {
          const auto& track = event_tracks[i_track];

          auto ids = track.ids();
          std::sort(std::begin(ids), std::end(ids));
          bool containsDuplicates = (std::unique(std::begin(ids), std::end(ids))) != std::end(ids);

          if (containsDuplicates) {
            warning_cout << "WARNING: Track #" << i_track << " contains duplicate LHCb IDs"
              << std::endl << std::hex;
            for (auto id : ids) {
              warning_cout << "0x" << id << ", ";
            }
            warning_cout << std::endl << std::endl << std::dec;
          } 
        }
      }
    }
  }
};
