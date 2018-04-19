#pragma once

#include "VeloDefinitions.cuh"

__device__ void fillCandidates(
  short* h0_candidates,
  short* h2_candidates,
  const unsigned int* module_hitStarts,
  const unsigned int* module_hitNums,
  const float* hit_Phis,
  const uint hit_offset
);