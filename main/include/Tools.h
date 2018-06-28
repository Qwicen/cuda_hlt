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
#include "../../cuda/velo/common/include/VeloDefinitions.cuh"
#include "../../cuda/velo/common/include/ClusteringDefinitions.cuh"
#include "../../cuda/veloUT/common/include/VeloUTDefinitions.cuh"
#include "../../checker/lib/include/Tracks.h"
#include "../include/InputTools.h"
#include "../../PrVeloUT/include/VeloTypes.h"

void readGeometry(
  const std::string& foldername,
  std::vector<char>& geometry
);

void check_velopix_events(
  const std::vector<char> events,
  const std::vector<unsigned int> event_offsets,
  int n_events
);

void read_ut_events_into_arrays(  VeloUTTracking::HitsSoA *ut_hits_events,
				  uint32_t n_hits_layers_events[][VeloUTTracking::n_layers],
				  const std::vector<char> events,
				  const std::vector<unsigned int> event_offsets,
				  int n_events );

void check_ut_events( const VeloUTTracking::HitsSoA *hits_layers_events,
		      const uint32_t n_hits_layers_events[][VeloUTTracking::n_layers],
		      const int n_events
		      );

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
  VeloTracking::Track <true> * host_tracks_pinned,
  int * host_accumulated_tracks,
  int * host_number_of_tracks_pinned,
  const int &number_of_events
);

trackChecker::Tracks prepareVeloUTTracks(
  std::vector< VeloUTTracking::TrackUT > ut_tracks
);

void call_pr_checker(
  const std::vector< trackChecker::Tracks >& all_tracks,
  const std::string& folder_name_MC,
  const bool& fromNtuple,
  const std::string& trackType
);
