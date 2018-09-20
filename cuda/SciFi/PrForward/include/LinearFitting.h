#pragma once

#include "SciFiDefinitions.cuh"
#include "TrackUtils.h"
#include "HitUtils.h"

#include <cmath>

void incrementLineFitParameters(
  SciFi::Tracking::LineFitterPars &parameters,
  SciFi::HitsSoA* hits_layers,
  float coordX[SciFi::Tracking::max_x_hits],
  int allXHits[SciFi::Tracking::max_x_hits],
  int it);

float getLineFitDistance(
  SciFi::Tracking::LineFitterPars &parameters,
  SciFi::HitsSoA* hits_layers,
  float coordX[SciFi::Tracking::max_x_hits],
  int allXHits[SciFi::Tracking::max_x_hits],
  int it );

float getLineFitChi2(
  SciFi::Tracking::LineFitterPars &parameters,
  SciFi::HitsSoA* hits_layers,
  float coordX[SciFi::Tracking::max_x_hits],
  int allXHits[SciFi::Tracking::max_x_hits],
  int it);

void solveLineFit(SciFi::Tracking::LineFitterPars &parameters);

void fastLinearFit(
  SciFi::HitsSoA* hits_layers,
  float trackParameters[SciFi::Tracking::nTrackParams],
  int coordToFit[SciFi::Tracking::max_coordToFit],
  int& n_coordToFit,
  PlaneCounter planeCounter,
  SciFi::Tracking::HitSearchCuts& pars_cur);
