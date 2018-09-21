#pragma once

#include <array>
#include <stdint.h>
#include "CudaCommon.h"
#include "VeloDefinitions.cuh"
#include "ClusteringDefinitions.cuh"
#include "ClusteringCommon.h"
#include "VeloUTDefinitions.cuh"
#include "PrForwardConstants.cuh"
#include "TMVA_Forward_1.cuh"
#include "TMVA_Forward_2.cuh"

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
  float host_ut_dxDy[VeloUTTracking::n_layers];
  SciFi::Tracking::TMVA* dev_scifi_tmva1;
  SciFi::Tracking::TMVA* dev_scifi_tmva2;
  SciFi::Tracking::Arrays* dev_scifi_constArrays;
  
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
