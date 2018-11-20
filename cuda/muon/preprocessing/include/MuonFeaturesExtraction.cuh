#pragma once

#include "Handler.cuh"
#include "MuonDefinitions.cuh"

__global__ void muon_catboost_features_extraction(
  const Muon::State* muTrack,
  const Muon::HitsSoA* muon_hits,
  float* dev_muon_catboost_features
);

ALGORITHM(muon_catboost_features_extraction, muon_catboost_features_extraction_t)