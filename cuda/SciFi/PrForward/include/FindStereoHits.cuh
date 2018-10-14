#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "SciFiDefinitions.cuh"
#include "TrackUtils.cuh"
#include "HitUtils.cuh"
#include "PrVeloUT.cuh"

/**
   Functions related to selecting hits on the uv planes,
   which match to the VeloUT input track
 */

__host__ __device__ void collectStereoHits(
  const SciFi::SciFiHits& scifi_hits,
  const SciFi::SciFiHitCount& scifi_hit_count,
  SciFi::Tracking::Track& track,
  MiniState velo_state,
  SciFi::Tracking::HitSearchCuts& pars,
  SciFi::Tracking::Arrays* constArrays,
  float stereoCoords[SciFi::Tracking::max_stereo_hits],
  int stereoHits[SciFi::Tracking::max_stereo_hits],
  int& n_stereoHits);

__host__ __device__ bool selectStereoHits(
  const SciFi::SciFiHits& scifi_hits,
  const SciFi::SciFiHitCount& scifi_hit_count,
  SciFi::Tracking::Track& track,
  SciFi::Tracking::Arrays* constArrays,
  float stereoCoords[SciFi::Tracking::max_stereo_hits],
  int stereoHits[SciFi::Tracking::max_stereo_hits],
  int& n_stereoHits,
  MiniState velo_state,
  SciFi::Tracking::HitSearchCuts& pars_cur);

__host__ __device__ bool addHitsOnEmptyStereoLayers(
  const SciFi::SciFiHits& scifi_hits,
  const SciFi::SciFiHitCount& scifi_hit_count,
  SciFi::Tracking::Track& track,
  int stereoHits[SciFi::Tracking::max_stereo_hits],
  int& n_stereoHits,
  SciFi::Tracking::Arrays* constArrays,
  PlaneCounter& planeCounter,
  MiniState velo_state,
  SciFi::Tracking::HitSearchCuts& pars_cur);
