#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include "SciFiDefinitions.cuh"
#include "TrackUtils.h"
#include "HitUtils.h"
#include "HoughTransform.h"

/**
   Functions related to selecting hits on the uv planes,
   which match to the VeloUT input track
 */

bool selectStereoHits(
  SciFi::HitsSoA* hits_layers,
  SciFi::Track& track,
  std::vector<int> stereoHits,
  FullState state_at_endvelo,
  SciFi::Tracking::HitSearchCuts& pars_cur);

bool addHitsOnEmptyStereoLayers(
  SciFi::HitsSoA* hits_layers,
  SciFi::Track& track,
  std::vector<int>& stereoHits,
  std::vector<unsigned int> &pc,
  int planelist[],
  FullState state_at_endvelo,
  SciFi::Tracking::HitSearchCuts& pars_cur);

std::vector<int> collectStereoHits(
  SciFi::HitsSoA* hits_layers,
  SciFi::Track& track,
  FullState state_at_endvelo,
  SciFi::Tracking::HitSearchCuts& pars_cur);
