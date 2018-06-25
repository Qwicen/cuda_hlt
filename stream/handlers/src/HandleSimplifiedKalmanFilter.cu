#include "HandleSimplifiedKalmanFilter.cuh"

void SimplifiedKalmanFilter::operator()() {
  velo_fit<<<num_blocks, num_threads, 0, *stream>>>(
    dev_velo_cluster_container,
    dev_module_cluster_start,
    dev_atomics_storage,
    dev_tracks,
    dev_velo_states
  );
}
