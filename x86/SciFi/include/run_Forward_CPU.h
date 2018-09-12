#pragma once

#include "Common.h"

#include "VeloDefinitions.cuh"
#include "TrackChecker.h"
#include "PrForward.h"
#include "Tools.h"

int run_forward_on_CPU (
  std::vector< trackChecker::Tracks >& ft_tracks_events,
  SciFi::HitsSoA * hits_layers_events,
  const VeloState * host_velo_states,
  const int * host_velo_accumulated_tracks,
  const VeloUTTracking::TrackUT * veloUT_tracks,
  const int * n_veloUT_tracks_events,
  const int &number_of_events
);
