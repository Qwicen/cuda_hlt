#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include "SciFiDefinitions.cuh"
#include "PrVeloUT.cuh"
#include "TrackUtils.cuh"

/**
   Project x hits onto reference plane
*/

__host__ __device__ void xAtRef_SamePlaneHits(
  const SciFi::SciFiHits& scifi_hits,
  const int allXHits[SciFi::Tracking::max_x_hits],
  const int n_x_hits,
  float coordX[SciFi::Tracking::max_x_hits],
  const float xParams_seed[4],
  SciFi::Tracking::Arrays* constArrays,
  MiniState velo_state,
  const float zMag, 
  int itH, int itEnd);
