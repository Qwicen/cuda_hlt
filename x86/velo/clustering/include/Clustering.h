#pragma once

#include "../../../../main/include/Common.h"
#include "../../../../main/include/Timer.h"
#include "../../../../main/include/Logger.h"
#include "../../../../cuda/velo/common/include/ClusteringDefinitions.cuh"
#include "ClusteringCommon.h"
#include <cstring>

constexpr unsigned int max_cluster_size = 196608;

std::vector<std::vector<uint32_t>> clustering(
  const std::vector<char>& geometry,
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets
);

std::vector<std::vector<uint32_t>> cuda_clustering(
  const std::vector<char>& geometry,
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets,
  const int verbosity = logger::error
);

std::vector<std::vector<uint32_t>> cuda_clustering_cpu_optimized(
  const std::vector<char>& geometry,
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets,
  const int verbosity = logger::error
);

std::vector<uint32_t> cuda_array_clustering(
  const std::vector<char>& geometry,
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets
);

void cache_sp_patterns(
  unsigned char* sp_patterns,
  unsigned char* sp_sizes,
  float* sp_fx,
  float* sp_fy
);
