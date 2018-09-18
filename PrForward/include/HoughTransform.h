#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include "TrackUtils.h"
#include "HitUtils.h"
#include "PrVeloUT.cuh"

/**
   Functions related to doing a 1D Hough transform
*/


void xAtRef_SamePlaneHits(
  SciFi::HitsSoA* hits_layers,
  std::vector<int>& allXHits,
  const float xParams_seed[4],
  MiniState velo_state, 
  int itH, int itEnd);

bool fitXProjection(
  SciFi::HitsSoA *hits_layers,
  std::vector<float> &trackParameters,
  std::vector<unsigned int> &coordToFit,
  PlaneCounter& planeCounter,
  SciFi::Tracking::HitSearchCuts& pars_cur);

bool fitYProjection(
  SciFi::HitsSoA *hits_layers,  
  SciFi::Track& track,
  std::vector<unsigned int>& stereoHits,
  PlaneCounter& planeCounter,
  MiniState velo_state,
  SciFi::Tracking::HitSearchCuts& pars_cur); 
