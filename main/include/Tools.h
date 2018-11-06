#pragma once

// #include <fstream>
// #include <numeric>

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

bool check_velopix_events(
  const std::vector<char>& events,
  const std::vector<uint>& event_offsets,
  size_t n_events
);

std::map<std::string, float> calcResults(
  std::vector<float>& times
);

void print_gpu_memory_consumption();

std::pair<size_t, std::string> set_device(
  int cuda_device
);
