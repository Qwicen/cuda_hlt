#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>
#include "SciFiDefinitions.cuh"
#include "SciFiEventModel.cuh"
#include "TrackUtils.cuh"
#include "HitUtils.cuh"
#include "PrVeloUT.cuh"

/**
   Functions related to selecting hits on the uv planes,
   which match to the VeloUT input track
 */

__host__ __device__ void collectStereoHits(
  const SciFi::Hits& scifi_hits,
  const SciFi::HitCount& scifi_hit_count,
  SciFi::Tracking::Track& track,
  const MiniState& velo_state,
  SciFi::Tracking::HitSearchCuts& pars,
  const SciFi::Tracking::Arrays* constArrays,
  float stereoCoords[SciFi::Tracking::max_stereo_hits],
  int stereoHits[SciFi::Tracking::max_stereo_hits],
  int& n_stereoHits);

__host__ __device__ bool selectStereoHits(
  const SciFi::Hits& scifi_hits,
  const SciFi::HitCount& scifi_hit_count,
  SciFi::Tracking::Track& track,
  const SciFi::Tracking::Arrays* constArrays,
  float stereoCoords[SciFi::Tracking::max_stereo_hits],
  int stereoHits[SciFi::Tracking::max_stereo_hits],
  int& n_stereoHits,
  const MiniState& velo_state,
  SciFi::Tracking::HitSearchCuts& pars_cur);

__host__ __device__ bool addHitsOnEmptyStereoLayers(
  const SciFi::Hits& scifi_hits,
  const SciFi::HitCount& scifi_hit_count,
  SciFi::Tracking::Track& track,
  int stereoHits[SciFi::Tracking::max_stereo_hits],
  int& n_stereoHits,
  const SciFi::Tracking::Arrays* constArrays,
  PlaneCounter& planeCounter,
  const MiniState& velo_state,
  SciFi::Tracking::HitSearchCuts& pars_cur);
