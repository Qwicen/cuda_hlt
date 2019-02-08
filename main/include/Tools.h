#pragma once

#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <cfloat>
#include <cstdint>
#include "CudaCommon.h"
#include "Logger.h"
#include "ClusteringDefinitions.cuh"
#include "MuonDefinitions.cuh"

bool check_velopix_events(const std::vector<char>& events, const std::vector<uint>& event_offsets, size_t n_events);

std::map<std::string, float> calcResults(std::vector<float>& times);

void print_gpu_memory_consumption();

std::pair<size_t, std::string> set_device(int cuda_device);

void read_muon_events_into_arrays(
  Muon::HitsSoA* muon_station_hits,
  const char* events,
  const uint* event_offsets,
  const int n_events);

void check_muon_events(const Muon::HitsSoA* muon_station_hits, const int hits_to_out, const int n_events);
