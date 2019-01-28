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
    bool* dev_is_muon,
    const Muon::Constants::FieldOfInterest* dev_muon_foi,
    const float* dev_muon_momentum_cuts
);

ALGORITHM(is_muon, is_muon_t)
