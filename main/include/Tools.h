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
#include "Tracks.h"
#include "InputTools.h"
#include "velopix-input-reader.h"
#include "TrackChecker.h"
#include "MCParticle.h"
#include "VeloConsolidated.cuh"

bool check_velopix_events(
  const std::vector<char> events,
  const std::vector<unsigned int> event_offsets,
  int n_events
);

// void check_ut_events(
//   const VeloUTTracking::HitsSoA *hits_layers_events,
//   const int n_events
// );

void read_UT_magnet_tool( PrUTMagnetTool* host_magnet_tool );

std::map<std::string, float> calcResults(
  std::vector<float>& times
);

std::vector<trackChecker::Tracks> prepareTracks(
  uint* host_velo_track_hit_number_pinned,
  char* host_velo_track_hits_pinned,
  int* host_accumulated_tracks,
  int* host_number_of_tracks_pinned,
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

void call_pr_checker(
  const std::vector< trackChecker::Tracks >& all_tracks,
  const std::string& folder_name_MC,
  const uint start_event_offset, 
  const std::string& trackType
);
