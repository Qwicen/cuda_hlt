#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include "TrackUtils.h"
#include "HitUtils.h"

void xAtRef_SamePlaneHits(
  ForwardTracking::HitsSoAFwd* hits_layers,
  std::vector<int>& allXHits,
  const float xParams_seed[4],
  FullState state_at_endvelo, 
  int itH, int itEnd);

bool fitXProjection(
  ForwardTracking::HitsSoAFwd *hits_layers,
  std::vector<float> &trackParameters,
  std::vector<unsigned int> &pc,
  int planelist[],
  PrParameters& pars_cur);

bool fitYProjection(
  ForwardTracking::HitsSoAFwd *hits_layers,  
  ForwardTracking::TrackForward& track,
  std::vector<int>& stereoHits,
  std::vector<unsigned int> &pc,
  int planelist[],
  FullState state_at_endvelo,
  PrParameters& pars_cur); 
