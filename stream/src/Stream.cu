#include "../include/Stream.cuh"

cudaError_t Stream::operator()(
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets,
  const std::vector<unsigned int>& hit_offsets,
  unsigned int start_event,
  unsigned int number_of_events,
  unsigned int number_of_repetitions
) {
  for (unsigned int repetitions=0; repetitions<number_of_repetitions; ++repetitions) {
    // Timers
    std::vector<std::pair<std::string, float>> times;
    Timer t_total;
    t_total.start();

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

    t.flush();
    t.start();

    // Copy required data
    cudaCheck(cudaMemcpy(dev_events, events.data(), events.size(), cudaMemcpyHostToDevice));

    t.stop();
    times.emplace_back("copy events", t.get());

    t.flush();
    t.start();

    cudaCheck(cudaMemcpy(dev_event_offsets, event_offsets.data(), event_offsets.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_hit_offsets, hit_offsets.data(), hit_offsets.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));

    t.stop();
    times.emplace_back("copy offsets", t.get());

    // Invoke kernel
    times.emplace_back(
      "calculatePhiAndSort",
      0.001 * Helper::invoke(calculatePhiAndSort)
    );

    cudaCheck(cudaPeekAtLastError());

    /////////////////////
    // SearchByTriplet //
    /////////////////////
    
    t.flush();
    t.start();

    // Initialize data
    cudaCheck(cudaMemset(dev_hit_used, false, total_number_of_hits * sizeof(bool)));
    cudaCheck(cudaMemset(dev_atomics_storage, 0, number_of_events * atomic_space * sizeof(int)));

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

    // TODO: The chain could follow from here on.
    // If the chain follows, we may not need to retrieve the data
    // in the state it is currently, but in a posterior state.
    // In principle, here we need to get back:
    // - dev_events: Shuffled input
    // - dev_hit_temp: hit Xs
    // - dev_atomics_storage: Number of tracks
    // - dev_tracks: Tracks
    // - dev_velo_states: VELO filtered states for each track
    
    // Therefore, this is just temporal
    // Free buffers

    // TODO: Fetch required data
    // std::vector<uint8_t> output (events.size());
    // std::vector<float> hit_xs (total_number_of_hits);
    // std::vector<char> consolidated_tracks (consolidated_tracks_size);
    // std::vector<VeloState> velo_states (total_number_of_tracks * STATES_PER_TRACK);

    // cudaCheck(cudaMemcpy(output.data(), dev_events, events.size(), cudaMemcpyDeviceToHost));
    // cudaCheck(cudaMemcpy(hit_xs.data(), dev_hit_temp, total_number_of_hits * sizeof(float), cudaMemcpyDeviceToHost));
    // cudaCheck(cudaMemcpy(consolidated_tracks.data(), dev_consolidated_tracks, consolidated_tracks_size, cudaMemcpyDeviceToHost));
    // cudaCheck(cudaMemcpy(velo_states.data(), dev_velo_states, total_number_of_tracks * STATES_PER_TRACK * sizeof(VeloState), cudaMemcpyDeviceToHost));

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
    << ", partial timers (ms): " << partial_times
    << std::endl;
}
