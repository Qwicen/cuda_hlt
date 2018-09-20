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


__host__ __device__ void xAtRef_SamePlaneHits(
  SciFi::HitsSoA* hits_layers,
  const int allXHits[SciFi::Tracking::max_x_hits],
  const int n_x_hits,
  float coordX[SciFi::Tracking::max_x_hits],
  const float xParams_seed[4],
  const SciFi::Tracking::Arrays& constArrays,
  MiniState velo_state, 
  int itH, int itEnd);

__host__ __device__ bool fitXProjection(
  SciFi::HitsSoA *hits_layers,
  float trackParameters[SciFi::Tracking::nTrackParams],
  int coordToFit[SciFi::Tracking::max_coordToFit],
  int& n_coordToFit,
  PlaneCounter& planeCounter,
  SciFi::Tracking::HitSearchCuts& pars_cur);

__host__ __device__ bool fitYProjection(
  SciFi::HitsSoA *hits_layers,  
  SciFi::Tracking::Track& track,
  int stereoHits[SciFi::Tracking::max_stereo_hits],
  int& n_stereoHits,
  PlaneCounter& planeCounter,
  MiniState velo_state,
  const SciFi::Tracking::Arrays& constArrays,
  SciFi::Tracking::HitSearchCuts& pars_cur); 

__host__ __device__ int fitParabola(
  int* coordToFit,
  const int n_coordToFit,
  SciFi::HitsSoA* hits_layers,
  float trackParameters[SciFi::Tracking::nTrackParams],
  const bool xFit);
