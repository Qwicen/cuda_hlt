#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include <cassert>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "Logger.h"
#include "SystemOfUnits.h"
#include "TMVA_Forward_1.cuh"
#include "TMVA_Forward_2.cuh"
#include "SciFiDefinitions.cuh"
#include "VeloDefinitions.cuh"
#include "VeloUTDefinitions.cuh"
#include "PrVeloUT.cuh"
#include "PrForwardConstants.cuh"
#include "TrackUtils.cuh"
#include "LinearFitting.cuh"
#include "HitUtils.cuh"
#include "FindXHits.cuh"
#include "FindStereoHits.cuh"
#include "VeloEventModel.cuh"
#include "VeloConsolidated.cuh"

__host__ __device__ void find_forward_tracks(
  const SciFi::SciFiHits& scifi_hits,
  const SciFi::SciFiHitCount& scifi_hit_count,
  const VeloUTTracking::TrackUT& veloUTTrack,
  SciFi::Track* outputTracks,
  uint* n_forward_tracks,
  SciFi::Tracking::TMVA* tmva1,
  SciFi::Tracking::TMVA* tmva2,
  SciFi::Tracking::Arrays* constArrays,
  const MiniState& velo_state);

__host__ __device__ void selectFullCandidates(
  const SciFi::SciFiHits& scifi_hits,
  const SciFi::SciFiHitCount& scifi_hit_count,
  SciFi::Tracking::Track* candidate_tracks,
  int& n_candidate_tracks,
  SciFi::Tracking::Track* selected_tracks,
  int& n_selected_tracks,
  const float xParams_seed[4],
  const float yParams_seed[4],
  MiniState velo_state,
  const float VeloUT_qOverP,
  SciFi::Tracking::HitSearchCuts& pars_cur,
  SciFi::Tracking::TMVA* tmva1,
  SciFi::Tracking::TMVA* tmva2,
  SciFi::Tracking::Arrays* constArrays,
  bool secondLoop);

__host__ __device__ SciFi::Track makeTrack( SciFi::Tracking::Track track ); 

template<class T>
__host__ __device__ void sort_tracks( 
  SciFi::Tracking::Track* tracks,
  const int n,
  const T& sort_function
) {
  // find permutations based on sort_function
  uint permutations[SciFi::Tracking::max_selected_tracks];
  for ( int i = 0; i < n; ++i ) {
    uint position = 0;
    for ( int j = 0; j < n; ++j ) {
      const int sort_result = sort_function( tracks[i], tracks[j] );
      position += sort_result>0 || (sort_result==0 && i>j);
    }
    permutations[position] = i;
  }

  // apply permutations, store tracks in temporary container
  SciFi::Tracking::Track tracks_tmp[SciFi::Tracking::max_selected_tracks];
  for ( int i = 0; i < n; ++i ) {
    const int index = permutations[i];
    tracks_tmp[i] = tracks[index];
  }
  
  // copy tracks back to original container
  for ( int i = 0; i < n; ++i ) {
    tracks[i] = tracks_tmp[i];
  }
}
