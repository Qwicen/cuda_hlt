#pragma once

#include <dirent.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <map>
#include <cmath>
#include <stdint.h>
#include "Logger.h"
#include "VeloDefinitions.cuh"
#include "ClusteringDefinitions.cuh"
#include "VeloUTDefinitions.cuh"
#include "PrVeloUTMagnetToolDefinitions.cuh"
#include "PrVeloUTDefinitions.cuh"
#include "SciFiDefinitions.cuh"
#include "Tracks.h"
#include "InputTools.h"
#include "velopix-input-reader.h"
#include "TrackChecker.h"
#include "MCParticle.h"

void readGeometry(
  const std::string& foldername,
  std::vector<char>& geometry
);

void check_velopix_events(
  const std::vector<char> events,
  const std::vector<unsigned int> event_offsets,
  int n_events
);

void read_scifi_events_into_arrays(  SciFi::HitsSoA *scifi_hits_events,
                                  uint32_t n_hits_layers_events[][SciFi::Constants::n_layers],
                                  const std::vector<char> events,
                                  const std::vector<unsigned int> event_offsets,
                                  int n_events );

void check_scifi_events( const SciFi::HitsSoA *hits_layers_events,
                      const uint32_t n_hits_layers_events[][SciFi::Constants::n_layers],
                      const int n_events
                      );

void read_ut_events_into_arrays(  VeloUTTracking::HitsSoA *ut_hits_events,
                                  const std::vector<char> events,
				  const std::vector<unsigned int> event_offsets,
				  int n_events );

void check_ut_events( const VeloUTTracking::HitsSoA *hits_layers_events,
                      const int n_events
		      );

void read_UT_magnet_tool( PrUTMagnetTool* host_magnet_tool );

std::map<std::string, float> calcResults(
  std::vector<float>& times
);

template <bool mc_check>
void printTrack(
  VeloTracking::Track<mc_check>* tracks,
  const int trackNumber,
  std::ofstream& outstream
);

template <bool mc_check>
void printTracks(
  VeloTracking::Track<mc_check>* tracks,
  int* n_tracks,
  int n_events,
  std::ofstream& outstream
);

template <bool mc_check>
void writeBinaryTrack(
  const unsigned int* hit_IDs,
  const VeloTracking::Track <mc_check> & track,
  std::ofstream& outstream
);

std::vector< trackChecker::Tracks > prepareTracks(
  uint* host_velo_track_hit_number_pinned,
  VeloTracking::Hit<true>* host_velo_track_hits_pinned,                                       
  int* host_accumulated_tracks,
  int* host_number_of_tracks_pinned,
  const int &number_of_events
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

trackChecker::Tracks prepareForwardTracks(
  std::vector< SciFi::Track > forward_tracks
);

void call_pr_checker(
  const std::vector< trackChecker::Tracks >& all_tracks,
  const std::string& folder_name_MC,
  const uint start_event_offset, 
  const std::string& trackType
);
