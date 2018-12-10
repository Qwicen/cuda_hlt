#pragma once

#include "Handler.cuh"
#include "MuonDefinitions.cuh"
#include "MiniState.cuh"

enum offset {
  DTS   = 0,
  TIMES = 1 * Muon::Constants::n_stations,
  CROSS = 2 * Muon::Constants::n_stations,
  RES_X = 3 * Muon::Constants::n_stations,
  RES_Y = 4 * Muon::Constants::n_stations
};

__global__ void muon_catboost_features_extraction(
  const MiniState* muTrack,
  const Muon::HitsSoA* muon_hits,
  const float* dev_scifi_qop,
  float* dev_muon_catboost_features
);

ALGORITHM(muon_catboost_features_extraction, muon_catboost_features_extraction_t)
