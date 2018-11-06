#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include "SciFiDefinitions.cuh"
#include "TrackUtils.cuh"

__host__ __device__ int fitParabola(
  int* coordToFit,
  const int n_coordToFit,
  const SciFi::SciFiHits& scifi_hits,
  float trackParameters[SciFi::Tracking::nTrackParams],
  const bool xFit);
