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
  const int * host_velo_number_of_tracks,
  std::vector< std::vector< VeloUTTracking::TrackUT > > ut_tracks,
  const int &number_of_events
);
