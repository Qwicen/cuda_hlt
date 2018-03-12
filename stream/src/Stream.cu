#include "../include/Stream.cuh"

cudaError_t Stream::operator()(
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets,
  const std::vector<unsigned int>& hit_offsets,
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
    const auto total_number_of_hits = hit_offsets[hit_offsets.size() - 1];

    /////////////////////////
    // CalculatePhiAndSort //
    /////////////////////////

    Timer t;

    if (dev_events_size < events.size()) {
      // malloc just this datatype
      cudaCheck(cudaFree(dev_events));
      dev_events_size = events.size();
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
      // Copy required data
      cudaCheck(cudaMemcpyAsync(dev_events, events.data(), events.size(), cudaMemcpyHostToDevice, stream));
      t.stop();
      times.emplace_back("copy events", t.get());
      
      t.restart();
      cudaCheck(cudaMemcpyAsync(dev_event_offsets, event_offsets.data(), event_offsets.size() * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
      cudaCheck(cudaMemcpyAsync(dev_hit_offsets, hit_offsets.data(), hit_offsets.size() * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
      t.stop();
      times.emplace_back("copy offsets", t.get());
    }

    // Invoke kernel
    times.emplace_back(
      "calculatePhiAndSort",
      0.001 * Helper::invoke(calculatePhiAndSort)
    );

    cudaCheck(cudaPeekAtLastError());

    /////////////////////
    // SearchByTriplet //
    /////////////////////
    
    t.restart();

    // Initialize data
    cudaCheck(cudaMemsetAsync(dev_hit_used, false, total_number_of_hits * sizeof(bool), stream));
    cudaCheck(cudaMemsetAsync(dev_atomics_storage, 0, number_of_events * atomic_space * sizeof(int), stream));

    t.stop();
    times.emplace_back("initialize sbt data", t.get());

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
    
    // Therefore, this is just temporal
    // Fetch required data
    if (transmit_device_to_host) {
      std::vector<int> number_of_tracks (number_of_events);
      std::vector<unsigned short> hit_permutations (total_number_of_hits);
      std::vector<Track> tracks (number_of_events * max_tracks_in_event);
      std::vector<VeloState> velo_states (number_of_events * max_tracks_in_event * STATES_PER_TRACK);

      cudaCheck(cudaMemcpyAsync(number_of_tracks.data(), dev_atomics_storage, number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(hit_permutations.data(), dev_hit_permutation, total_number_of_hits * sizeof(unsigned short), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(tracks.data(), dev_tracks, number_of_events * max_tracks_in_event * sizeof(Track), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(velo_states.data(), dev_velo_states, number_of_events * max_tracks_in_event * STATES_PER_TRACK * sizeof(VeloState), cudaMemcpyDeviceToHost, stream));
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
