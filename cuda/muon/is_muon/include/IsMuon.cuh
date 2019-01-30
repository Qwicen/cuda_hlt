#pragma once

#include "Handler.cuh"
#include "MuonDefinitions.cuh"
#include "MiniState.cuh"

__global__ void is_muon(
  int* dev_atomics_scifi,
  uint* dev_scifi_track_hit_number,
  float* dev_scifi_qop,
  MiniState* dev_scifi_states,
  uint* dev_scifi_track_ut_indices,
  const Muon::HitsSoA* muon_hits,
  bool* dev_is_muon,
  const Muon::Constants::FieldOfInterest* dev_muon_foi,
  const float* dev_muon_momentum_cuts
);

__device__ std::pair<float,float> field_of_interest(
  const Muon::Constants::FieldOfInterest* dev_muon_foi,
  const int station,
  const int region,
  const float momentum
);

__device__ bool is_in_window(
  const float hit_x,
  const float hit_y,
  const float hit_dx,
  const float hit_dy,
  const Muon::Constants::FieldOfInterest* dev_muon_foi,
  const int station,
  const int region,
  const float p,
  const float extrapolation_x,
  const float extrapolation_y
);

ALGORITHM(is_muon, is_muon_t)
