#pragma once

#include "Common.h"
#include "Handler.cuh"
#include "PV_Definitions.cuh"
#include "VeloEventModel.cuh"
#include "VeloConsolidated.cuh"

__global__ void ip_cut(
  char* dev_kalman_velo_states,
  int* dev_atomics_velo,
  uint* dev_velo_track_hit_number,
  char* dev_velo_pv_ip,
  bool* dev_accepted_velo_tracks);

ALGORITHM(ip_cut, ip_cut_t)
