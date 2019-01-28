#pragma once

#include <array>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include "CudaCommon.h"
#include "VeloDefinitions.cuh"
#include "ClusteringDefinitions.cuh"
#include "ClusteringCommon.h"
#include "PrForwardConstants.cuh"
#include "TMVA_Forward_1.cuh"
#include "TMVA_Forward_2.cuh"
#include "UTDefinitions.cuh"
#include "Logger.h"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "KalmanParametrizations.cuh"

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
  std::array<float, UT::Constants::n_layers> host_ut_dxDy;
  std::array<uint, UT::Constants::n_layers + 1> host_unique_x_sector_layer_offsets;
  std::vector<uint> host_unique_x_sector_offsets;
  std::vector<float> host_unique_sector_xs;
  std::array<uint, UT::Constants::n_layers * UT::Constants::n_regions_in_layer + 1> host_ut_region_offsets;
  std::array<uint8_t, VeloClustering::lookup_table_size> host_candidate_ks;
  std::array<float, 9> host_inv_clus_res;

  float* dev_velo_module_zs;
  uint8_t* dev_velo_candidate_ks;
  uint8_t* dev_velo_sp_patterns;
  float* dev_velo_sp_fx;
  float* dev_velo_sp_fy;
  float* dev_ut_dxDy;
  SciFi::Tracking::TMVA* dev_scifi_tmva1;
  SciFi::Tracking::TMVA* dev_scifi_tmva2;
  SciFi::Tracking::Arrays* dev_scifi_constArrays;
  uint* dev_unique_x_sector_layer_offsets;
  uint* dev_unique_x_sector_offsets;
  uint* dev_ut_region_offsets;
  float* dev_unique_sector_xs;
  float* dev_inv_clus_res;

  // Geometry constants
  char* dev_velo_geometry;
  char* dev_ut_boards;
  char* dev_ut_geometry;
  char* dev_scifi_geometry;
  const char* host_scifi_geometry; 
  PrUTMagnetTool* dev_ut_magnet_tool;
  
  // Muon classification model constatns
  int muon_catboost_n_features;
  int muon_catboost_n_trees;
  int* dev_muon_catboost_tree_depths;
  int* dev_muon_catboost_tree_offsets;
  int* dev_muon_catboost_split_features;
  float* dev_muon_catboost_split_borders;
  float* dev_muon_catboost_leaf_values;
  int* dev_muon_catboost_leaf_offsets;

  // Kalman filter.
  ParKalmanFilter::KalmanParametrizations* dev_kalman_params;

  /**
   * @brief Reserves and initializes constants.
   */
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

  /**
   * @brief Initializes UT decoding constants.
   */
  void initialize_ut_decoding_constants(const std::vector<char>& ut_geometry);

  /**
   * @brief Initializes geometry constants and magnet field.
   */
  void initialize_geometry_constants(
    const std::vector<char>& velopix_geometry,
    const std::vector<char>& ut_boards,
    const std::vector<char>& ut_geometry,
    const std::vector<char>& ut_magnet_tool,
    const std::vector<char>& scifi_geometry);

  void initialize_muon_catboost_model_constants(
    const int n_features,
    const int n_trees,
    const std::vector<int>& tree_depths,
    const std::vector<int>& tree_offsets,
    const std::vector<float>& leaf_values,
    const std::vector<int>& leaf_offsets,
    const std::vector<float>& split_borders,
    const std::vector<int>& split_features
  );

};
