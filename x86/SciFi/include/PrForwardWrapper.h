#pragma once

#include "PrForward.cuh"

void PrForwardWrapper(
  const SciFi::SciFiHits& hits_layers,
  const Velo::Consolidated::States& velo_states,
  const uint event_tracks_offset,
  const VeloUTTracking::TrackUT * veloUT_tracks,
  const int n_veloUT_tracks,
  SciFi::Tracking::TMVA* tmva1,
  SciFi::Tracking::TMVA* tmva2,
  SciFi::Tracking::Arrays* constArrays,
  SciFi::Track outputTracks[SciFi::max_tracks],
  uint* n_forward_tracks);
