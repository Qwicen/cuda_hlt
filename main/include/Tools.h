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
#include "../../checker/lib/include/Tracks.h"

bool naturalOrder(
  const std::string& s1,
  const std::string& s2
);

void readFileIntoVector(
  const std::string& filename,
  std::vector<char>& events
);

void appendFileToVector(
  const std::string& filename,
  std::vector<char>& events,
  std::vector<unsigned int>& event_sizes
);

void readGeometry(
  const std::string& foldername,
  std::vector<char>& geometry
);

void check_events(
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets,
  int n_events
);

std::vector<std::string> list_folder(
  const std::string& foldername,
  const bool print_num_elems = false
);

void read_folder(
  const std::string& foldername,
  unsigned int fileNumber,
  std::vector<char>& events,
  std::vector<unsigned int>& event_offsets
);

void read_mc_folder(
  const std::string& foldername,
  unsigned int fileNumber,
  std::vector<char>& events,
  std::vector<unsigned int>& event_offsets
);

std::map<std::string, float> calcResults(
  std::vector<float>& times
);

template <bool mc_check>
void printTrack(
  Track<mc_check>* tracks,
  const int trackNumber,
  std::ofstream& outstream
);

template <bool mc_check>
void printTracks(
  Track<mc_check>* tracks,
  int* n_tracks,
  int n_events,
  std::ofstream& outstream
);

template <bool mc_check>
void writeBinaryTrack(
  const unsigned int* hit_IDs,
  const Track <mc_check> & track,
  std::ofstream& outstream
);

void call_PrChecker(
  const std::vector<trackChecker::Tracks>& all_tracks,
  const std::string& folder_name_MC
);

void checkTracks(
  int* host_number_of_tracks_pinned,
  int* host_accumulated_tracks,
  uint* host_velo_track_hit_number_pinned,
  Hit<true>* host_velo_track_hits_pinned,
  const int number_of_events,
  const std::string& folder_name_MC
);
