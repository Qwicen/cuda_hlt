#pragma once

#include <stdint.h>
#include "../../common/include/VeloDefinitions.cuh"

__device__ float velo_kalman_filter_step(
  const float z,
  const float zhit,
  const float xhit,
  const float whit,
  float& x,
  float& tx,
  float& covXX,
  float& covXTx,
  float& covTxTx
);

template<bool mc_check_enabled>
__global__ void velo_fit(
  const uint32_t* dev_velo_cluster_container,
  const uint* dev_module_cluster_start,
  const int* dev_atomics_storage,
  const Track<mc_check_enabled>* dev_tracks,
  VeloState* dev_velo_states
);

#include "VeloKalmanFilter_impl.cuh"
