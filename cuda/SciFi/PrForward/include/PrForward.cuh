#pragma once

#include "PrForwardTools.cuh"
#include "Handler.cuh"
// #include "ArgumentsCommon.cuh"
#include "ArgumentsVelo.cuh"
#include "ArgumentsUT.cuh"
#include "ArgumentsSciFi.cuh"

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
  uint32_t* dev_scifi_hits,
  const uint32_t* dev_scifi_hit_count,
  const int* dev_atomics_velo,
  const uint* dev_velo_track_hit_number,
  const char* dev_velo_states,
  const int* dev_atomics_ut,
  const char* dev_ut_track_hits,
  const uint* dev_ut_track_hit_number,
  const float* dev_ut_qop,
  const uint* dev_ut_track_velo_indices,
  SciFi::TrackHits* dev_scifi_tracks,
  int* dev_atomics_scifi,
  const SciFi::Tracking::TMVA* dev_tmva1,
  const SciFi::Tracking::TMVA* dev_tmva2,
  const SciFi::Tracking::Arrays* dev_constArrays,
  const char* dev_scifi_geometry,
  const float* dev_inv_clus_res);

ALGORITHM(
  scifi_pr_forward,
  scifi_pr_forward_t,
  ARGUMENTS(
    dev_scifi_hits,
    dev_scifi_hit_count,
    dev_atomics_velo,
    dev_velo_track_hit_number,
    dev_velo_states,
    dev_atomics_ut,
    dev_ut_track_hits,
    dev_ut_track_hit_number,
    dev_ut_qop,
    dev_ut_track_velo_indices,
    dev_scifi_tracks,
    dev_atomics_scifi))
