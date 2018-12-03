#pragma once

#include "Handler.cuh"
#include "MuonDefinitions.cuh"
#include "MiniState.cuh"


__global__ void muon_catboost_features_extraction(
  const MiniState* muTrack,
  const Muon::HitsSoA* muon_hits,
  const float* dev_scifi_qop,
  float* dev_muon_catboost_features
);

ALGORITHM(muon_catboost_features_extraction, muon_catboost_features_extraction_t)
