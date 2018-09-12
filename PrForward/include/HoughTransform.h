#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include "TrackUtils.h"
#include "HitUtils.h"

/**
   Functions related to doing a 1D Hough transform
*/


void xAtRef_SamePlaneHits(
  SciFi::Constants::HitsSoAFwd* hits_layers,
  std::vector<int>& allXHits,
  const float xParams_seed[4],
  FullState state_at_endvelo, 
  int itH, int itEnd);

bool fitXProjection(
  SciFi::Constants::HitsSoAFwd *hits_layers,
  std::vector<float> &trackParameters,
  std::vector<unsigned int> &pc,
  int planelist[],
  PrParameters& pars_cur);

bool fitYProjection(
  SciFi::Constants::HitsSoAFwd *hits_layers,  
  SciFi::Constants::TrackForward& track,
  std::vector<int>& stereoHits,
  std::vector<unsigned int> &pc,
  int planelist[],
  FullState state_at_endvelo,
  PrParameters& pars_cur); 
