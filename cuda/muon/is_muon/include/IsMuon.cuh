#pragma once

#include "Handler.cuh"
#include "MuonDefinitions.cuh"
#include "MiniState.cuh"
#include "ArgumentsSciFi.cuh"
#include "ArgumentsMuon.cuh"

__global__ void is_muon(
  int* dev_atomics_scifi,
  uint* dev_scifi_track_hit_number,
  float* dev_scifi_qop,
  MiniState* dev_scifi_states,
  uint* dev_scifi_track_ut_indices,
  const Muon::HitsSoA* muon_hits,
  int* dev_muon_track_occupancies,
  bool* dev_is_muon,
  const uint* event_list,
  const Muon::Constants::FieldOfInterest* dev_muon_foi,
  const float* dev_muon_momentum_cuts
);

ALGORITHM(  
  is_muon, 
  is_muon_t,
  ARGUMENTS(
    dev_atomics_scifi,
    dev_scifi_track_hit_number,
    dev_scifi_qop,
    dev_scifi_states,
    dev_scifi_track_ut_indices,
    dev_muon_hits,
    dev_muon_track_occupancies,
    dev_is_muon,
    dev_event_list))
