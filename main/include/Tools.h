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

/**
 * Generic StrException launcher
 */

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

void check_velopix_events( const std::vector<char> events,
		   const std::vector<unsigned int> event_offsets,
		   int n_events
		   );

void read_ut_events_into_arrays(  VeloUTTracking::Hits * ut_hits_events[][VeloUTTracking::n_layers],
				  uint32_t * n_hits_layers_events[][VeloUTTracking::n_layers],
				  const std::vector<char> events,
				  const std::vector<unsigned int> event_offsets,
				  int n_events );

void check_ut_events( const std::vector<char> events,
		      const std::vector<unsigned int> event_offsets,
		      int n_events
		      );

void readFolder(
  const std::string& foldername,
  unsigned int fileNumber,
  std::vector<char>& events,
  std::vector<unsigned int>& event_offsets
);

void statistics(
  const std::vector<char>& input,
  std::vector<unsigned int>& event_offsets
);

std::map<std::string, float> calcResults(
  std::vector<float>& times
);

void printOutSensorHits(
  const EventInfo& info,
  int sensorNumber,
  int* prevs,
  int* nexts
);

void printOutAllSensorHits(
  const EventInfo& info,
  int* prevs,
  int* nexts
);

void printInfo(
  const EventInfo& info,
  int numberOfSensors,
  int numberOfHits
);

template <bool mc_check>
void printTrack(
  Track <mc_check> * tracks,
  const int trackNumber,
  std::ofstream& outstream
);

template <bool mc_check>
void printTracks(
  Track <mc_check> * tracks,
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

cudaError_t checkSorting(
  const std::vector<std::vector<uint8_t>>& input,
  unsigned int acc_hits,
  unsigned short* dev_hit_phi,
  const std::vector<unsigned int>& hit_offsets
);

void call_PrChecker(
		 const std::vector< trackChecker::Tracks > all_tracks,
		 const std::string folder_name_MC
);

void checkTracks(
		 Track <do_mc_check> * host_tracks_pinned,
		 int * host_accumulated_tracks,
		 int * host_number_of_tracks_pinned,
		 const int &number_of_events,
		 const std::string folder_name_MC
);
