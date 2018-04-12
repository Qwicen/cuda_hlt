#include "../include/Stream.cuh"

cudaError_t Stream::operator()(
  const char* host_events_pinned,
  const uint* host_event_offsets_pinned,
  const uint* host_hit_offsets_pinned,
  size_t host_events_pinned_size,
  size_t host_event_offsets_pinned_size,
  size_t host_hit_offsets_pinned_size,
  uint start_event,
  uint number_of_events,
  uint number_of_repetitions
) {
  for (uint repetitions=0; repetitions<number_of_repetitions; ++repetitions) {
    ////////////////
    // Clustering //
    ////////////////

    if (transmit_host_to_device) {
      cudaCheck(cudaMemcpyAsync(dev_raw_input, host_events_pinned, host_events_pinned_size, cudaMemcpyHostToDevice, stream));
      cudaCheck(cudaMemcpyAsync(dev_raw_input_offsets, host_event_offsets_pinned, host_event_offsets_pinned_size * sizeof(uint), cudaMemcpyHostToDevice, stream));
    }

    // Estimate the input size of each module
    estimateInputSize();

    // Convert the estimated sizes to module hit start format (offsets)
    prefixSum();

    // // Fetch the number of hits we require
    // uint number_of_hits;
    // cudaCheck(cudaMemcpyAsync(&number_of_hits, dev_estimated_input_size + number_of_events * 52, sizeof(uint), cudaMemcpyDeviceToHost, stream));

    // if (number_of_hits * 6 * sizeof(uint32_t) > velo_cluster_container_size) {
    //   WARNING << "Number of hits: " << number_of_hits << std::endl
    //     << "Size of velo cluster container is larger than previously accomodated." << std::endl
    //     << "Resizing from " << velo_cluster_container_size << " to " << number_of_hits * 6 * sizeof(uint) << " B" << std::endl;

    //   cudaCheck(cudaFree(dev_velo_cluster_container));
    //   velo_cluster_container_size = number_of_hits * 6 * sizeof(uint32_t);
    //   cudaCheck(cudaMalloc((void**)&dev_velo_cluster_container, velo_cluster_container_size));
    // }

    // Invoke clustering
    maskedVeloClustering();

    /////////////////////////
    // CalculatePhiAndSort //
    /////////////////////////

    // Invoke sorting
    calculatePhiAndSort();

    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    // /////////////////////
    // // SearchByTriplet //
    // /////////////////////

    // searchByTriplet();

    // ////////////////////////
    // // Consolidate tracks //
    // ////////////////////////
    
    // consolidateTracks();

    // // Optional transmission device to host
    // if (transmit_device_to_host) {
    //   cudaCheck(cudaMemcpyAsync(host_number_of_tracks_pinned, dev_atomics_storage, number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
    //   cudaEventRecord(cuda_generic_event, stream);
    //   cudaEventSynchronize(cuda_generic_event);
      
    //   int total_number_of_tracks = 0;
    //   for (int i=0; i<number_of_events; ++i) {
    //     total_number_of_tracks += host_number_of_tracks_pinned[i];
    //   }

    //   cudaCheck(cudaMemcpyAsync(host_tracks_pinned, dev_tracklets, total_number_of_tracks * sizeof(Track), cudaMemcpyDeviceToHost, stream));
    // }
  }

  return cudaSuccess;
}
