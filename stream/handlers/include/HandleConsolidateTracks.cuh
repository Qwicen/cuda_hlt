#pragma once

#include "../../../cuda/velo/common/include/VeloDefinitions.cuh"
#include "Handler.cuh"

struct ConsolidateTracks : public Handler {
  // Call parameters
  int* dev_atomics_storage;
  VeloTracking::TrackHits* dev_tracks;
  VeloTracking::Track <mc_check_enabled> * dev_output_tracks;
  uint32_t* dev_velo_cluster_container;
  uint* dev_module_cluster_start;
  uint* dev_module_cluster_num;
  VeloState* dev_velo_states;

  ConsolidateTracks() = default;

  void setParameters(
    int* param_dev_atomics_storage,
    VeloTracking::TrackHits* param_dev_tracks,
    VeloTracking::Track <mc_check_enabled> * param_dev_output_tracks,
    uint32_t* param_dev_velo_cluster_container,
    uint* param_dev_module_cluster_start,
    uint* param_dev_module_cluster_num,
    VeloState* param_dev_velo_states
  ) {
    dev_atomics_storage = param_dev_atomics_storage;
    dev_tracks = param_dev_tracks;
    dev_output_tracks = param_dev_output_tracks;
    dev_velo_cluster_container = param_dev_velo_cluster_container;
    dev_module_cluster_start = param_dev_module_cluster_start;
    dev_module_cluster_num =  param_dev_module_cluster_num;
    dev_velo_states = param_dev_velo_states;
  }

  void operator()();
};
