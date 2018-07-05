#pragma once
#include "../include/Stream.cuh"
#include "../../../main/include/Common.h"

#include "../../../PrVeloUT/src/PrVeloUT.h"

int run_veloUT_on_CPU (
  std::vector< trackChecker::Tracks > * ut_tracks_events,
  const VeloUTTracking::HitsSoA * hits_layers_events,
  const uint32_t n_hits_layers_events[][VeloUTTracking::n_layers],
  VeloState * host_velo_states,
  int * host_accumulated_tracks,
  VeloTracking::Track <mc_check_enabled> *host_tracks_pinned,
  int * host_number_of_tracks_pinned,
  const int &number_of_events
);
