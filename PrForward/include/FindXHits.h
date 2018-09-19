#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "SciFiDefinitions.cuh"
#include "PrForwardConstants.h"
#include "VeloUTDefinitions.cuh"
#include "TrackUtils.h"
#include "HitUtils.h"
#include "LinearFitting.h"
#include "HoughTransform.h"
#include "PrVeloUT.cuh"

/**
   Functions related to selecting hits on the x planes,
   which match to the VeloUT input track
 */

void collectAllXHits(
  SciFi::HitsSoA* hits_layers,
  int allXHits[SciFi::Tracking::max_x_hits],
  int& n_x_hits,
  const float xParams_seed[4],
  const float yParams_seed[4],
  const MiniState& velo_state,
  const float qop,
  int side);

void selectXCandidates(
  SciFi::HitsSoA* hits_layers,
  int allXHits[SciFi::Tracking::max_x_hits],
  int& n_x_hits,
  const VeloUTTracking::TrackUT& veloUTTrack,
  SciFi::Track candidate_tracks[SciFi::max_tracks],
  int& n_candidate_tracks,
  const float zRef_track,
  const float xParams_seed[4],
  const float yParams_seed[4],
  const MiniState& velo_state,
  SciFi::Tracking::HitSearchCuts& pars_cur,
  int side);

bool addHitsOnEmptyXLayers(
  SciFi::HitsSoA* hits_layers,
  std::vector<float> &trackParameters,
  const float xParams_seed[4],
  const float yParams_seed[4],
  bool fullFit,
  std::vector<unsigned int> &coordToFit,
  PlaneCounter& planeCounter,
  SciFi::Tracking::HitSearchCuts& pars_cur,
  int side);