#pragma once

#include "../../../cuda/velo/consolidate_tracks/include/ConsolidateTracks.cuh"
#include "Handler.cuh"

struct ConsolidateTracks : public Handler {
  // Call parameters
  int* dev_atomics_storage;
  VeloTracking::TrackHits* dev_tracks;
  VeloTracking::Track <do_mc_check> * dev_output_tracks;
  VeloState* dev_velo_states;
  VeloState* dev_velo_states_out;
  uint32_t* dev_velo_cluster_container;
  uint* dev_module_cluster_start;
  uint* dev_module_cluster_num;
  
  ConsolidateTracks() = default;

  void setParameters(
    int* param_dev_atomics_storage,
    VeloTracking::TrackHits* param_dev_tracks,
    VeloTracking::Track <do_mc_check> * param_dev_output_tracks,
    VeloState* param_dev_velo_states,
    VeloState* param_dev_velo_states_out,
    uint32_t* param_dev_velo_cluster_container,
    uint* param_dev_module_cluster_start,
    uint* param_dev_module_cluster_num
  ) {
    dev_atomics_storage = param_dev_atomics_storage;
    dev_tracks = param_dev_tracks;
    dev_output_tracks = param_dev_output_tracks;
    dev_velo_states = param_dev_velo_states;
    dev_velo_states_out = param_dev_velo_states_out;
    dev_velo_cluster_container = param_dev_velo_cluster_container;
    dev_module_cluster_start = param_dev_module_cluster_start;
    dev_module_cluster_num =  param_dev_module_cluster_num;

  }

   void operator()();
};
