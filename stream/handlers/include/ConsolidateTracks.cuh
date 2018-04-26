#pragma once

#include "../../../cuda/velo/consolidate_tracks/include/ConsolidateTracks.cuh"
#include "Handler.cuh"

struct ConsolidateTracks : public Handler {
  // Call parameters
  int* dev_atomics_storage;
  Track* dev_tracks;
  Track* dev_output_tracks;

  ConsolidateTracks() = default;

  void setParameters(
    int* param_dev_atomics_storage,
    Track* param_dev_tracks,
    Track* param_dev_output_tracks
  ) {
    dev_atomics_storage = param_dev_atomics_storage;
    dev_tracks = param_dev_tracks;
    dev_output_tracks = param_dev_output_tracks;
  }

  void operator()();
};
