#pragma once

#include "PrForwardTools.cuh"

void PrForwardWrapper(
  const SciFi::Hits& scifi_hits,
  const SciFi::HitCount& scifi_hit_count,
  const Velo::Consolidated::States& velo_states,
  const uint event_tracks_offset,
  const UT::Consolidated::Tracks& ut_tracks,
  const int n_veloUT_tracks,
  const SciFi::Tracking::TMVA* tmva1,
  const SciFi::Tracking::TMVA* tmva2,
  const SciFi::Tracking::Arrays* constArrays,
  SciFi::TrackHits outputTracks[SciFi::Constants::max_tracks],
  uint* n_forward_tracks);
