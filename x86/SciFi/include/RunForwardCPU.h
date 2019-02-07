#pragma once

#include "Common.h"
#include "VeloDefinitions.cuh"
#include "TrackChecker.h"
#include "PrForwardWrapper.h"
#include "VeloEventModel.cuh"
#include "CpuHandler.cuh"
#include "PrepareTracks.h"
#include "ArgumentsCommon.cuh"
#include "ArgumentsVelo.cuh"
#include "ArgumentsUT.cuh"
#include "ArgumentsSciFi.cuh"

int run_forward_on_CPU (
  SciFi::TrackHits* host_scifi_tracks_events,
  int* host_scifi_n_tracks,
  const uint* host_scifi_hits,
  const uint* host_scifi_hit_count,
  const char* host_scifi_geometry, 
  const std::array<float, 9>& host_inv_clus_res,
  const uint* host_velo_tracks_atomics,
  const uint* host_velo_track_hit_number,
  const char* host_velo_states,
  const int * host_atomics_ut,
  const uint* host_ut_track_hit_number,
  const float* host_ut_qop,
  const uint* host_ut_track_velo_indices,
  const uint number_of_events);

CPU_ALGORITHM(run_forward_on_CPU, cpu_scifi_pr_forward_t,
  ARGUMENTS(dev_scifi_tracks,
    dev_atomics_scifi,
    dev_scifi_hits,
    dev_scifi_hit_count,
    dev_atomics_velo,
    dev_velo_track_hit_number,
    dev_velo_states,
    dev_atomics_ut,
    dev_ut_track_hits,
    dev_ut_track_hit_number,
    dev_ut_qop,
    dev_ut_track_velo_indices))
