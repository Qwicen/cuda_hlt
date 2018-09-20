#pragma once

#include "SciFiDefinitions.cuh"
#include "TrackUtils.cuh"
#include "HitUtils.cuh"

#include <cmath>

__host__ __device__ void incrementLineFitParameters(
  SciFi::Tracking::LineFitterPars &parameters,
  SciFi::HitsSoA* hits_layers,
  float coordX[SciFi::Tracking::max_x_hits],
  int allXHits[SciFi::Tracking::max_x_hits],
  int it);

__host__ __device__ float getLineFitDistance(
  SciFi::Tracking::LineFitterPars &parameters,
  SciFi::HitsSoA* hits_layers,
  float coordX[SciFi::Tracking::max_x_hits],
  int allXHits[SciFi::Tracking::max_x_hits],
  int it );

__host__ __device__ float getLineFitChi2(
  SciFi::Tracking::LineFitterPars &parameters,
  SciFi::HitsSoA* hits_layers,
  float coordX[SciFi::Tracking::max_x_hits],
  int allXHits[SciFi::Tracking::max_x_hits],
  int it);

__host__ __device__ void solveLineFit(SciFi::Tracking::LineFitterPars &parameters);

__host__ __device__ void fastLinearFit(
  SciFi::HitsSoA* hits_layers,
  float trackParameters[SciFi::Tracking::nTrackParams],
  int coordToFit[SciFi::Tracking::max_coordToFit],
  int& n_coordToFit,
  PlaneCounter planeCounter,
  SciFi::Tracking::HitSearchCuts& pars_cur);
