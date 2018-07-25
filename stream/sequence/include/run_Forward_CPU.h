#pragma once

#include "../../../main/include/Common.h"

#include "../../../checker/lib/include/TrackChecker.h"
#include "../../../PrForward/src/PrForward.h"
#include "../../../main/include/Tools.h"

std::vector< std::vector< VeloUTTracking::TrackVeloUT > > run_forward_on_CPU (
  std::vector< trackChecker::Tracks > * ft_tracks_events,
  ForwardTracking::HitsSoAFwd * hits_layers_events,
  const uint32_t n_hits_layers_events[][ForwardTracking::n_layers],
  std::vector< std::vector< VeloUTTracking::TrackVeloUT > > ut_tracks,
  const int &number_of_events
);
