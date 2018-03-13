#include "../include/Stream.cuh"

cudaError_t Stream::operator()(
  const char* host_events_pinned,
  const unsigned int* host_event_offsets_pinned,
  const unsigned int* host_hit_offsets_pinned,
  size_t host_events_pinned_size,
  size_t host_event_offsets_pinned_size,
  size_t host_hit_offsets_pinned_size,
  unsigned int start_event,
  unsigned int number_of_events,
  unsigned int number_of_repetitions
) {
  cudaCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
  
  for (unsigned int repetitions=0; repetitions<number_of_repetitions; ++repetitions) {
    // Timers
    std::vector<std::pair<std::string, float>> times;
    Timer t_total;

    // Total number of hits
    const auto total_number_of_hits = host_hit_offsets_pinned[host_hit_offsets_pinned_size - 1];

    ////////////////////////////
    // Calculate phi and sort //
    ////////////////////////////

    Timer t;

    if (dev_events_size < host_events_pinned_size) {
      // malloc just this datatype
      cudaCheck(cudaFree(dev_events));
      dev_events_size = host_events_pinned_size;
      cudaCheck(cudaMalloc((void**)&dev_events, dev_events_size));
    }

    if ((total_number_of_hits / number_of_events) > maximum_average_number_of_hits_per_event) {
      std::cerr << "total average number of hits exceeds maximum ("
        << (total_number_of_hits / number_of_events) << " > " << maximum_average_number_of_hits_per_event
        << ")" << std::endl;
    }

    t.stop();
    times.emplace_back("allocate phi buffers", t.get());

    if (transmit_host_to_device) {
      t.restart();
      cudaCheck(cudaMemcpyAsync(dev_events, host_events_pinned, host_events_pinned_size, cudaMemcpyHostToDevice, stream));
      cudaCheck(cudaMemcpyAsync(dev_event_offsets, host_event_offsets_pinned, host_event_offsets_pinned_size * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
      cudaCheck(cudaMemcpyAsync(dev_hit_offsets, host_hit_offsets_pinned, host_hit_offsets_pinned_size * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
      t.stop();
      times.emplace_back("copy input events", t.get());
    }

    // Invoke kernel
    times.emplace_back(
      "calculatePhiAndSort",
      0.001 * Helper::invoke(calculatePhiAndSort)
    );

    cudaCheck(cudaPeekAtLastError());

    ///////////////////////
    // Search by triplet //
    ///////////////////////
    
    // Invoke kernel
    times.emplace_back(
      "Search by triplets",
      0.001 * Helper::invoke(searchByTriplet)
    );
    cudaCheck(cudaPeekAtLastError());

    ///////////////////////////
    // Calculate VELO states //
    ///////////////////////////

    // Invoke kernel
    times.emplace_back(
      "calculateVeloStates",
      0.001 * Helper::invoke(calculateVeloStates)
    );
    cudaCheck(cudaPeekAtLastError());

    // The chain can follow from here on.
    // If the chain follows, we may not need to retrieve the data
    // in the state it is currently, but in a posterior state.
    // In principle, here we need to get back:
    // - dev_hit_permutation: Permutation of hits (reorder)
    // - dev_atomics_storage: Number of tracks
    // - dev_tracks: Tracks
    // - dev_velo_states: VELO filtered states for each track
    
    // This transmission back is unoptimized
    // (ie. consolidation can happen prior to VELO execution)
    // As the GPU HLT1 evolves, so will this excerpt of code
    if (transmit_device_to_host) {
      times.emplace_back(
        "consolidateTracks",
        0.001 * Helper::invoke(consolidateTracks)
      );
      cudaCheck(cudaPeekAtLastError());

      cudaCheck(cudaMemcpyAsync(host_number_of_tracks_pinned, dev_atomics_storage, number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
      cudaEventRecord(cuda_generic_event, stream);
      cudaEventSynchronize(cuda_generic_event);

      int total_number_of_tracks = 0;
      for (int i=0; i<number_of_events; ++i) {
        total_number_of_tracks += host_number_of_tracks_pinned[i];
      }

      // Consolidated tracks are currently stored in dev_tracklets
      cudaCheck(cudaMemcpyAsync(host_tracks_pinned, dev_tracklets, total_number_of_tracks * sizeof(Track), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_velo_states, dev_velo_states, number_of_events * max_tracks_in_event * STATES_PER_TRACK * sizeof(VeloState), cudaMemcpyDeviceToHost, stream));
    }

    t_total.stop();
    times.emplace_back("total", t_total.get());

    if (do_print_timing) {
      print_timing(number_of_events, times);
    }
  }

  return cudaSuccess;
}

void Stream::print_timing(
  const unsigned int number_of_events,
  const std::vector<std::pair<std::string, float>>& times
) {
  const auto total_time = times[times.size() - 1];
  std::string partial_times = "{\n";
  for (size_t i=0; i<times.size(); ++i) {
    if (i != times.size()-1) {
      partial_times += " " + times[i].first + "\t" + std::to_string(times[i].second) + "\t("
        + std::to_string(100 * (times[i].second / total_time.second)) + " %)\n";
    } else {
      partial_times += " " + times[i].first + "\t" + std::to_string(times[i].second) + "\t("
        + std::to_string(100 * (times[i].second / total_time.second)) + " %)\n}";
    }
  }

  DEBUG << "stream #" << stream_number << ": "
    << number_of_events / total_time.second << " events/s"
    << ", partial timers (s): " << partial_times
    << std::endl;
}
