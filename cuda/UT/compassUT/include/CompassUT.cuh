#pragma once

#include "PrVeloUT.cuh"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "UTDefinitions.cuh"
#include "VeloDefinitions.cuh"
#include "CompassUTDefinitions.cuh"
#include "FindBestHits.cuh"
#include "Handler.cuh"
#include "ArgumentsCommon.cuh"
#include "ArgumentsVelo.cuh"
#include "ArgumentsUT.cuh"
#include "UTEventModel.cuh"

//=========================================================================
// Functions definitions
//=========================================================================
__global__ void compass_ut(
  uint* dev_ut_hits,
  const uint* dev_ut_hit_offsets,
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  char* dev_velo_track_hits,
  char* dev_velo_states,
  PrUTMagnetTool* dev_ut_magnet_tool,
  const float* dev_ut_dxDy,
  int* dev_active_tracks,
  const uint* dev_unique_x_sector_layer_offsets,
  const float* dev_unique_sector_xs,
  UT::TrackHits* dev_compassUT_tracks,
  int* dev_atomics_compassUT,
  short* dev_windows_layers,
  bool* dev_accepted_velo_tracks);

__device__ void compass_ut_tracking(
  const short* dev_windows_layers,
  char* dev_velo_track_hits,
  const uint number_of_tracks_event,
  const int i_track,
  const uint current_track_offset,
  const Velo::Consolidated::States& velo_states,
  const Velo::Consolidated::Tracks& velo_tracks,
  const UT::Hits& ut_hits,
  const UT::HitOffsets& ut_hit_offsets,
  const float* bdl_table,
  const float* dev_ut_dxDy,
  short* win_size_shared,
  int* n_veloUT_tracks_event,
  UT::TrackHits* veloUT_tracks_event,
  const int event_hit_offset);

__host__ __device__ __inline__ bool velo_track_in_UT_acceptance(
  const MiniState& state);

__device__ __inline__ void fill_shared_windows(
  const short* windows_layers,
  const int number_of_tracks_event,
  const int i_track,
  short* win_size_shared);

__device__ __inline__ bool found_active_windows(
  const short* dev_windows_layers,
  const int total_tracks_event,
  const int track);

__device__ void save_track(
  const int i_track,
  const float* bdlTable,
  const MiniState& velo_state,
  const BestParams& best_params,
  char* dev_velo_track_hits,
  const Velo::Consolidated::Tracks& velo_tracks,
  const int num_best_hits,
  const int* best_hits,
  const UT::Hits& ut_hits,
  const float* ut_dxDy,
  int* n_veloUT_tracks,
  UT::TrackHits* VeloUT_tracks,
  const int event_hit_offset);

ALGORITHM(compass_ut, compass_ut_t,
  ARGUMENTS(dev_ut_hits,
    dev_ut_hit_offsets,
    dev_atomics_velo,
    dev_velo_track_hit_number,
    dev_velo_track_hits,
    dev_velo_states,
    dev_ut_tracks,
    dev_atomics_ut,
    dev_ut_active_tracks,
    dev_ut_windows_layers,
    dev_accepted_velo_tracks))
