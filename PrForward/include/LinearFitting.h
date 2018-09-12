#pragma once

#include "SciFiDefinitions.cuh"
#include "TrackUtils.h"
#include "HitUtils.h"


void incrementLineFitParameters(SciFi::Constants::LineFitterPars &parameters, SciFi::Constants::HitsSoAFwd* hits_layers, int it);

float getLineFitDistance(SciFi::Constants::LineFitterPars &parameters, SciFi::Constants::HitsSoAFwd* hits_layers, int it );

float getLineFitChi2(SciFi::Constants::LineFitterPars &parameters, SciFi::Constants::HitsSoAFwd* hits_layers, int it);

void solveLineFit(SciFi::Constants::LineFitterPars &parameters);

void fastLinearFit(
  SciFi::Constants::HitsSoAFwd* hits_layers,
  std::vector<float> &trackParameters,
  std::vector<unsigned int> &pc,
  int planelist[],
  PrParameters& pars_cur);
