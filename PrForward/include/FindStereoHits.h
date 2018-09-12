#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include "ForwardDefinitions.cuh"
#include "TrackUtils.h"
#include "HitUtils.h"
#include "HoughTransform.h"

bool selectStereoHits(
  ForwardTracking::HitsSoAFwd* hits_layers,
  ForwardTracking::TrackForward& track,
  std::vector<int> stereoHits,
  FullState state_at_endvelo,
  PrParameters& pars_cur);

bool addHitsOnEmptyStereoLayers(
  ForwardTracking::HitsSoAFwd* hits_layers,
  ForwardTracking::TrackForward& track,
  std::vector<int>& stereoHits,
  std::vector<unsigned int> &pc,
  int planelist[],
  FullState state_at_endvelo,
  PrParameters& pars_cur);

std::vector<int> collectStereoHits(
  ForwardTracking::HitsSoAFwd* hits_layers,
  ForwardTracking::TrackForward& track,
  FullState state_at_endvelo,
  PrParameters& pars_cur);
