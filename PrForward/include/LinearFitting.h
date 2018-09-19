#pragma once

#include "SciFiDefinitions.cuh"
#include "TrackUtils.h"
#include "HitUtils.h"

#include <cmath>

void incrementLineFitParameters(SciFi::Tracking::LineFitterPars &parameters, SciFi::HitsSoA* hits_layers, int it);

float getLineFitDistance(SciFi::Tracking::LineFitterPars &parameters, SciFi::HitsSoA* hits_layers, int it );

float getLineFitChi2(SciFi::Tracking::LineFitterPars &parameters, SciFi::HitsSoA* hits_layers, int it);

void solveLineFit(SciFi::Tracking::LineFitterPars &parameters);

void fastLinearFit(
  SciFi::HitsSoA* hits_layers,
  float trackParameters[SciFi::Tracking::nTrackParams],
  std::vector<unsigned int> &coordToFit,
  PlaneCounter planeCounter,
  SciFi::Tracking::HitSearchCuts& pars_cur);
