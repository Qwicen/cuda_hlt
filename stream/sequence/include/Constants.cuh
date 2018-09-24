#pragma once

#include <array>
#include <stdint.h>
#include "CudaCommon.h"
#include "VeloDefinitions.cuh"
#include "ClusteringDefinitions.cuh"
#include "ClusteringCommon.h"
#include "VeloUTDefinitions.cuh"

/**
 * @brief Struct intended as a singleton with constants defined on GPU.
 * @details __constant__ memory on the GPU has very few use cases.
 *          Instead, global memory is preferred. Hence, this singleton
 *          should allocate the requested buffers on GPU and serve the
 *          pointers wherever needed.
 *          
 *          The pointers are hard-coded. Feel free to write more as needed.
 */
struct Constants {
  float* dev_velo_module_zs;
  uint8_t* dev_velo_candidate_ks;
  uint8_t* dev_velo_sp_patterns;
  float* dev_velo_sp_fx;
  float* dev_velo_sp_fy;
  float* dev_ut_dxDy;
  
  std::array<uint8_t, 9> host_candidate_ks;
  float host_ut_dxDy[VeloUTTracking::n_layers];
  
  void reserve_and_initialize() {
    reserve_constants();
    initialize_constants();
  }

  /**
   * @brief Reserves the constants of the GPU.
   */
  void reserve_constants();

  /**
   * @brief Initializes constants on the GPU.
   */
  void initialize_constants();
};
