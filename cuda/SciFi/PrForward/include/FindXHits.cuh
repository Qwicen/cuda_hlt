#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>
#include "SciFiDefinitions.cuh"
#include "PrForwardConstants.cuh"
#include "UTDefinitions.cuh"
#include "TrackUtils.cuh"
#include "HitUtils.cuh"
#include "LinearFitting.cuh"
#include "ReferencePlaneProjection.cuh"
#include "PrVeloUT.cuh"
#include "SciFiEventModel.cuh"

/**
   Functions related to selecting hits on the x planes,
   which match to the VeloUT input track
 */

__host__ __device__ void collectAllXHits(
  const SciFi::Hits& scifi_hits,
  const SciFi::HitCount& scifi_hit_count,
  int allXHits[SciFi::Tracking::max_x_hits],
  int& n_x_hits,
  float coordX[SciFi::Tracking::max_x_hits],
  const float xParams_seed[4],
  const float yParams_seed[4],
  const SciFi::Tracking::Arrays* constArrays,
  const MiniState& velo_state,
  const float qop,
  int side);

__host__ __device__ void improveXCluster(
  int& it2,
  const int it1,
  const int itEnd,
  const int n_x_hits,
  const bool usedHits[SciFi::Tracking::max_x_hits],
  const float coordX[SciFi::Tracking::max_x_hits],
  const float xWindow,
  const SciFi::Tracking::HitSearchCuts& pars,
  PlaneCounter& planeCounter,
  const int allXHits[SciFi::Tracking::max_x_hits],
  const SciFi::Hits& scifi_hits);

__host__ __device__ void selectXCandidates(
  const SciFi::Hits& scifi_hits,
  const SciFi::HitCount& scifi_hit_count,
  int allXHits[SciFi::Tracking::max_x_hits],
  int& n_x_hits,
  bool usedHits[SciFi::Tracking::max_x_hits],
  float coordX[SciFi::Tracking::max_x_hits],
  SciFi::Tracking::Track candidate_tracks[SciFi::Constants::max_tracks],
  int& n_candidate_tracks,
  const float zRef_track,
  const float xParams_seed[4],
  const float yParams_seed[4],
  const MiniState& velo_state,
  SciFi::Tracking::HitSearchCuts& pars,
  const SciFi::Tracking::Arrays* constArrays,
  int side,
  const bool secondLoop);

__host__ __device__ bool addHitsOnEmptyXLayers(
  const SciFi::Hits& scifi_hits,
  const SciFi::HitCount& scifi_hit_count,
  float trackParameters[SciFi::Tracking::nTrackParams],
  const float xParams_seed[4],
  const float yParams_seed[4],
  bool fullFit,
  int coordToFit[SciFi::Tracking::max_coordToFit],
  int& n_coordToFit,
  const SciFi::Tracking::Arrays* constArrays,
  PlaneCounter& planeCounter,
  SciFi::Tracking::HitSearchCuts& pars_cur,
  int side);
