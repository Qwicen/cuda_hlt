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
#include "HoughTransform.cuh"
#include "FindXHits.cuh"
#include "FindStereoHits.cuh"
#include "VeloEventModel.cuh"
#include "VeloConsolidated.cuh"

/** @class PrForward PrForward.h
   *
   *  - InputTracksName: Input location for VeloUT tracks
   *  - OutputTracksName: Output location for Forward tracks
   *  Based on code written by
   *  2012-03-20 : Olivier Callot
   *  2013-03-15 : Thomas Nikodem
   *  2015-02-13 : Sevda Esen [additional search in the triangles by Marian Stahl]
   *  2016-03-09 : Thomas Nikodem [complete restructuring]
   */

__global__ void PrForward(
  SciFi::HitsSoA* dev_scifi_hits,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  uint* dev_velo_states,
  VeloUTTracking::TrackUT * dev_veloUT_tracks,
  const int* dev_atomics_veloUT,
  SciFi::Track* dev_scifi_tracks,
  uint* dev_n_scifi_tracks ,
  SciFi::Tracking::TMVA* dev_tmva1,
  SciFi::Tracking::TMVA* dev_tmva2,
  SciFi::Tracking::Arrays* dev_constArrays);

__host__ __device__ void find_forward_tracks(
  SciFi::HitsSoA* hits_layers,  
  const VeloUTTracking::TrackUT& veloUTTrack,
  SciFi::Track* outputTracks,
  uint* n_forward_tracks,
  SciFi::Tracking::TMVA* tmva1,
  SciFi::Tracking::TMVA* tmva2,
  SciFi::Tracking::Arrays* constArrays,
  const MiniState& velo_state);


__host__ __device__ void selectFullCandidates(
  SciFi::HitsSoA* hits_layers,
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
