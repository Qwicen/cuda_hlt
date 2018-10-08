#pragma once

#include <cfloat>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <map>
#include <cmath>
#include <cstdint>
#include "Logger.h"
#include "VeloEventModel.cuh"
#include "ClusteringDefinitions.cuh"
#include "VeloUTDefinitions.cuh"
#include "SciFiDefinitions.cuh"
#include "Tracks.h"
#include "InputTools.h"
#include "velopix-input-reader.h"
#include "TrackChecker.h"
#include "MCParticle.h"
#include "VeloConsolidated.cuh"

bool check_velopix_events(
  const std::vector<char>& events,
  const std::vector<uint>& event_offsets,
  int n_events
);

std::map<std::string, float> calcResults(
  std::vector<float>& times
);

std::vector<trackChecker::Tracks> prepareTracks(
  uint* host_velo_tracks_atomics,
  uint* host_velo_track_hit_number_pinned,
  char* host_velo_track_hits_pinned,
  const uint number_of_events
);

trackChecker::Tracks prepareVeloUTTracksEvent(
  const VeloUTTracking::TrackUT* veloUT_tracks,
  const int n_veloUT_tracks
);

std::vector< trackChecker::Tracks > prepareVeloUTTracks(
  const VeloUTTracking::TrackUT* veloUT_tracks,
  const int* n_veloUT_tracks,
  const int number_of_events
);

trackChecker::Tracks prepareForwardTracksVeloUTOnly(
  std::vector< VeloUTTracking::TrackUT > forward_tracks
); 

trackChecker::Tracks prepareForwardTracksEvent(
  SciFi::Track forward_tracks[SciFi::max_tracks],
  const uint n_forward_tracks
);

std::vector< trackChecker::Tracks > prepareForwardTracks(
  SciFi::Track* scifi_tracks,
  uint* n_scifi_tracks,
  const int number_of_events
);

void call_pr_checker(
  const std::vector< trackChecker::Tracks >& all_tracks,
  const std::string& folder_name_MC,
  const uint start_event_offset, 
  const std::string& trackType
);
