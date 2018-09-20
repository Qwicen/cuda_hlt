#pragma once

#include "PrForward.cuh"

void PrForwardWrapper(
  SciFi::HitsSoA *hits_layers,
  const Velo::Consolidated::States& velo_states,
  const uint event_tracks_offset,
  const VeloUTTracking::TrackUT * veloUT_tracks,
  const int n_veloUT_tracks,
  const SciFi::Tracking::TMVA& tmva1,
  const SciFi::Tracking::TMVA& tmva2,
  const SciFi::Tracking::Arrays& constArrays,
  SciFi::Track outputTracks[SciFi::max_tracks],
  int& n_forward_tracks);
