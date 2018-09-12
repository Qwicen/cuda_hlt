#pragma once

#include "ForwardDefinitions.cuh"
#include "TrackUtils.h"
#include "HitUtils.h"


void incrementLineFitParameters(ForwardTracking::LineFitterPars &parameters, ForwardTracking::HitsSoAFwd* hits_layers, int it);

float getLineFitDistance(ForwardTracking::LineFitterPars &parameters, ForwardTracking::HitsSoAFwd* hits_layers, int it );

float getLineFitChi2(ForwardTracking::LineFitterPars &parameters, ForwardTracking::HitsSoAFwd* hits_layers, int it);

void solveLineFit(ForwardTracking::LineFitterPars &parameters);

void fastLinearFit(
  ForwardTracking::HitsSoAFwd* hits_layers,
  std::vector<float> &trackParameters,
  std::vector<unsigned int> &pc,
  int planelist[],
  PrParameters& pars_cur);
