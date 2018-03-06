#include "../include/Stream.cuh"
#include "../include/CalculatePhiAndSort.cuh"
#include "../include/SearchByTriplet.cuh"
#include "../include/CalculateVeloStates.cuh"
#include "../include/Helper.cuh"

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

    // Number of defined atomics
    constexpr unsigned int atomic_space = NUM_ATOMICS + 1;

    // Total number of hits
    const auto total_number_of_hits = hit_offsets[hit_offsets.size() - 1];
    DEBUG << (total_number_of_hits / number_of_events) << " average hits per event" << std::endl;

    // Blocks and threads for each algorithm
    dim3 num_blocks (number_of_events);
    dim3 sort_num_threads (64);
    dim3 sbt_num_threads (NUMTHREADS_X);
    dim3 velo_states_num_threads (1024);

    // Datatypes
    Track* dev_tracks;
    char* dev_events;
    unsigned int* dev_tracks_to_follow;
    bool* dev_hit_used;
    int* dev_atomics_storage;
    Track* dev_tracklets;
    unsigned int* dev_weak_tracks;
    unsigned int* dev_event_offsets;
    unsigned int* dev_hit_offsets;
    short* dev_h0_candidates;
    short* dev_h2_candidates;
    unsigned short* dev_rel_indices;
    float* dev_hit_phi;
    int32_t* dev_hit_temp;
    unsigned short* dev_hit_permutation;
    VeloState* dev_velo_states;

    /////////////////////////
    // CalculatePhiAndSort //
    /////////////////////////

    Timer t;

    // Allocate buffers
    cudaCheck(cudaMalloc((void**)&dev_events, events.size()));
    cudaCheck(cudaMalloc((void**)&dev_event_offsets, event_offsets.size() * sizeof(unsigned int)));
    cudaCheck(cudaMalloc((void**)&dev_hit_offsets, hit_offsets.size() * sizeof(unsigned int)));
    cudaCheck(cudaMalloc((void**)&dev_hit_phi, total_number_of_hits * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dev_hit_temp, total_number_of_hits * sizeof(int32_t)));
    cudaCheck(cudaMalloc((void**)&dev_hit_permutation, total_number_of_hits * sizeof(unsigned short)));

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
    auto calculatePhiAndSort = CalculatePhiAndSort(
      num_blocks,
      sort_num_threads,
      stream,
      dev_events,
      dev_event_offsets,
      dev_hit_offsets,
      dev_hit_phi,
      dev_hit_temp,
      dev_hit_permutation
    );

    times.emplace_back(
      "calculatePhiAndSort",
      0.001 * Helper::invoke(calculatePhiAndSort)
    );

    cudaCheck(cudaPeekAtLastError());

    // Free buffers
    cudaCheck(cudaFree(dev_hit_permutation));

    /////////////////////
    // SearchByTriplet //
    /////////////////////

    t.flush();
    t.start();

    // Allocate buffers
    cudaCheck(cudaMalloc((void**)&dev_tracks, number_of_events * MAX_TRACKS * sizeof(Track)));
    cudaCheck(cudaMalloc((void**)&dev_tracks_to_follow, number_of_events * TTF_MODULO * sizeof(unsigned int)));
    cudaCheck(cudaMalloc((void**)&dev_hit_used, total_number_of_hits * sizeof(bool)));
    cudaCheck(cudaMalloc((void**)&dev_atomics_storage, number_of_events * atomic_space * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&dev_tracklets, total_number_of_hits * sizeof(Track)));
    cudaCheck(cudaMalloc((void**)&dev_weak_tracks, total_number_of_hits * sizeof(unsigned int)));
    cudaCheck(cudaMalloc((void**)&dev_h0_candidates, 2 * total_number_of_hits * sizeof(short)));
    cudaCheck(cudaMalloc((void**)&dev_h2_candidates, 2 * total_number_of_hits * sizeof(short)));
    cudaCheck(cudaMalloc((void**)&dev_rel_indices, number_of_events * MAX_NUMHITS_IN_MODULE * sizeof(unsigned short)));
    
    t.stop();
    times.emplace_back("allocate sbt buffers", t.get());

    t.flush();
    t.start();

    // Initialize data
    cudaCheck(cudaMemset(dev_hit_used, false, total_number_of_hits * sizeof(bool)));
    cudaCheck(cudaMemset(dev_atomics_storage, 0, number_of_events * atomic_space * sizeof(int)));

    t.stop();
    times.emplace_back("initialize sbt data", t.get());

    // Invoke kernel
    auto searchByTriplet = SearchByTriplet(
      num_blocks,
      sbt_num_threads,
      stream,
      dev_tracks,
      dev_events,
      dev_tracks_to_follow,
      dev_hit_used,
      dev_atomics_storage,
      dev_tracklets,
      dev_weak_tracks,
      dev_event_offsets,
      dev_hit_offsets,
      dev_h0_candidates,
      dev_h2_candidates,
      dev_rel_indices,
      dev_hit_phi,
      dev_hit_temp
    );

    times.emplace_back(
      "sbt",
      0.001 * Helper::invoke(searchByTriplet)
    );

    cudaCheck(cudaPeekAtLastError());

    // Free buffers
    cudaCheck(cudaFree(dev_tracks_to_follow));
    cudaCheck(cudaFree(dev_hit_used));
    cudaCheck(cudaFree(dev_tracklets));
    cudaCheck(cudaFree(dev_weak_tracks));
    cudaCheck(cudaFree(dev_h0_candidates));
    cudaCheck(cudaFree(dev_h2_candidates));
    cudaCheck(cudaFree(dev_rel_indices));

    ///////////////////////////
    // Calculate VELO states //
    ///////////////////////////

    // Allocate buffers
    cudaCheck(cudaMalloc((void**)&dev_velo_states, number_of_events * MAX_TRACKS * STATES_PER_TRACK * sizeof(VeloState)));

    // Invoke kernel
    auto calculateVeloStates = CalculateVeloStates(
      num_blocks,
      velo_states_num_threads,
      stream,
      dev_events,
      dev_atomics_storage,
      dev_tracks,
      dev_velo_states,
      dev_hit_temp,
      dev_event_offsets,
      dev_hit_offsets
    );

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
    cudaCheck(cudaFree(dev_event_offsets));
    cudaCheck(cudaFree(dev_hit_offsets));

    // TODO: Fetch required data
    // std::vector<uint8_t> output (events.size());
    // std::vector<float> hit_xs (total_number_of_hits);
    // std::vector<char> consolidated_tracks (consolidated_tracks_size);
    // std::vector<VeloState> velo_states (total_number_of_tracks * STATES_PER_TRACK);

    // cudaCheck(cudaMemcpy(output.data(), dev_events, events.size(), cudaMemcpyDeviceToHost));
    // cudaCheck(cudaMemcpy(hit_xs.data(), dev_hit_temp, total_number_of_hits * sizeof(float), cudaMemcpyDeviceToHost));
    // cudaCheck(cudaMemcpy(consolidated_tracks.data(), dev_consolidated_tracks, consolidated_tracks_size, cudaMemcpyDeviceToHost));
    // cudaCheck(cudaMemcpy(velo_states.data(), dev_velo_states, total_number_of_tracks * STATES_PER_TRACK * sizeof(VeloState), cudaMemcpyDeviceToHost));

    // Free buffers
    cudaCheck(cudaFree(dev_events));
    cudaCheck(cudaFree(dev_hit_temp));
    cudaCheck(cudaFree(dev_atomics_storage));
    cudaCheck(cudaFree(dev_tracks));
    cudaCheck(cudaFree(dev_velo_states));

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
