#pragma once

#include "PrForwardTools.cuh"
#include "Handler.cuh"

/** @class PrForward PrForward.h
   *
   *  - InputTracksName: Input location for VeloUT tracks
   *  - OutputTracksName: Output location for Forward tracks
   *  Based on code written by
   *  2012-03-20 : Olivier Callot
   *  2013-03-15 : Thomas Nikodem
   *  2015-02-13 : Sevda Esen [additional search in the triangles by Marian Stahl]
   *  2016-03-09 : Thomas Nikodem [complete restructuring]
   *  2018-08    : Vava Gligorov [extract code from Rec, make compile within GPU framework
   *  2018-09    : Dorothea vom Bruch [convert to CUDA, runs on GPU]
   */

__global__ void scifi_pr_forward(
  const uint* dev_scifi_hits,
  const uint32_t* dev_scifi_hit_count,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  uint* dev_velo_states,
  VeloUTTracking::TrackUT * dev_veloUT_tracks,
  const int * dev_atomics_veloUT,
  SciFi::Track* dev_scifi_tracks,
  uint* dev_n_scifi_tracks ,
  SciFi::Tracking::TMVA* dev_tmva1,
  SciFi::Tracking::TMVA* dev_tmva2,
  SciFi::Tracking::Arrays* dev_constArrays);

ALGORITHM(scifi_pr_forward, scifi_pr_forward_t)
