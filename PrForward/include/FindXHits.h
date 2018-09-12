#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include "SciFiDefinitions.cuh"
#include "PrForwardConstants.h"
#include "VeloUTDefinitions.cuh"
#include "TrackUtils.h"
#include "HitUtils.h"
#include "LinearFitting.h"
#include "HoughTransform.h"

/**
   Functions related to selecting hits on the x planes,
   which match to the VeloUT input track
 */

void collectAllXHits(
  SciFi::Constants::HitsSoAFwd* hits_layers,
  std::vector<int>& allXHits, 
  const float xParams_seed[4],
  const float yParams_seed[4],
  FullState state_at_endvelo,
  int side);

void selectXCandidates(
  SciFi::Constants::HitsSoAFwd* hits_layers,
  std::vector<int>& allXHits,
  const VeloUTTracking::TrackVeloUT& veloUTTrack,
  std::vector<SciFi::Constants::TrackForward>& outputTracks,
  const float zRef_track,
  const float xParams_seed[4],
  const float yParams_seed[4],
  FullState state_at_endvelo,
  PrParameters& pars_cur,
  int side);

bool addHitsOnEmptyXLayers(
  SciFi::Constants::HitsSoAFwd* hits_layers,
  std::vector<float> &trackParameters,
  const float xParams_seed[4],
  const float yParams_seed[4],
  bool fullFit,
  std::vector<unsigned int> &pc,
  int planelist[],
  PrParameters& pars_cur,
  int side);
