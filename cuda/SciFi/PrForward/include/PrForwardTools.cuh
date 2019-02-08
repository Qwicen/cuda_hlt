#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include <cassert>
#include "Logger.h"
#include "SystemOfUnits.h"
#include "TMVA_Forward_1.cuh"
#include "TMVA_Forward_2.cuh"
#include "SciFiDefinitions.cuh"
#include "VeloDefinitions.cuh"
#include "UTDefinitions.cuh"
#include "PrVeloUT.cuh"
#include "PrForwardConstants.cuh"
#include "TrackUtils.cuh"
#include "LinearFitting.cuh"
#include "HitUtils.cuh"
#include "FindXHits.cuh"
#include "FindStereoHits.cuh"
#include "VeloEventModel.cuh"
#include "VeloConsolidated.cuh"
#include "UTConsolidated.cuh"
#include "SciFiEventModel.cuh"
#include "MiniState.cuh"

__host__ __device__ void find_forward_tracks(
  const SciFi::Hits& scifi_hits,
  const SciFi::HitCount& scifi_hit_count,
  const float qop_ut,
  const int i_veloUT_track,
  SciFi::TrackHits* outputTracks,
  uint* n_forward_tracks,
  const SciFi::Tracking::TMVA* tmva1,
  const SciFi::Tracking::TMVA* tmva2,
  const SciFi::Tracking::Arrays* constArrays,
  const MiniState& velo_state);

__host__ __device__ void selectFullCandidates(
  const SciFi::Hits& scifi_hits,
  const SciFi::HitCount& scifi_hit_count,
  SciFi::Tracking::Track* candidate_tracks,
  int& n_candidate_tracks,
  SciFi::Tracking::Track* selected_tracks,
  int& n_selected_tracks,
  const float xParams_seed[4],
  const float yParams_seed[4],
  MiniState velo_state,
  const float VeloUT_qOverP,
  SciFi::Tracking::HitSearchCuts& pars_cur,
  const SciFi::Tracking::TMVA* tmva1,
  const SciFi::Tracking::TMVA* tmva2,
  const SciFi::Tracking::Arrays* constArrays,
  const bool secondLoop);

__host__ __device__ SciFi::TrackHits makeTrack(SciFi::Tracking::Track track);

template<class T>
__host__ __device__ void sort_tracks(SciFi::Tracking::Track* tracks, const int n, const T& sort_function)
{
  // find permutations based on sort_function
  uint permutations[SciFi::Tracking::max_selected_tracks];
  for (int i = 0; i < n; ++i) {
    uint position = 0;
    for (int j = 0; j < n; ++j) {
      const int sort_result = sort_function(tracks[i], tracks[j]);
      position += sort_result > 0 || (sort_result == 0 && i > j);
    }
    permutations[position] = i;
  }

  // apply permutations, store tracks in temporary container
  SciFi::Tracking::Track tracks_tmp[SciFi::Tracking::max_selected_tracks];
  for (int i = 0; i < n; ++i) {
    const int index = permutations[i];
    tracks_tmp[i] = tracks[index];
  }

  // copy tracks back to original container
  for (int i = 0; i < n; ++i) {
    tracks[i] = tracks_tmp[i];
  }
}
