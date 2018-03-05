#include "../include/Stream.cuh"

cudaError_t Stream::operator()(
  const std::vector<std::vector<uint8_t>>& input,
  unsigned int start_event,
  unsigned int number_of_events
) {
  // Number of defined atomics
  constexpr unsigned int atomic_space = NUM_ATOMICS + 1;

  // Blocks and threads for each algorithm
  dim3 num_blocks (number_of_events);
  dim3 sort_num_threads (64);
  dim3 sbt_num_threads (NUMTHREADS_X);
  dim3 velo_states_num_threads (1024);

  // Timer
  std::vector<float> times;

  // Prepare event offset and hit offset
  std::vector<unsigned int> event_offsets;
  std::vector<unsigned int> hit_offsets;
  unsigned int acc_size=0, acc_hits=0;
  for (unsigned int i=0; i<number_of_events; ++i) {
    const auto event_no = start_event + i;
    auto info = EventInfo(input[event_no]);
    const int event_size = input[event_no].size();
    event_offsets.push_back(acc_size);
    hit_offsets.push_back(acc_hits);
    acc_size += event_size;
    acc_hits += info.numberOfHits;
  }

  // Datatypes
  Track* dev_tracks;
  char* dev_input;
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
  char* dev_consolidated_tracks;
  VeloState* dev_velo_states;

  /////////////////////////
  // CalculatePhiAndSort //
  /////////////////////////

  // Allocate buffers
  cudaCheck(cudaMalloc((void**)&dev_input, acc_size));
  cudaCheck(cudaMalloc((void**)&dev_event_offsets, event_offsets.size() * sizeof(unsigned int)));
  cudaCheck(cudaMalloc((void**)&dev_hit_offsets, hit_offsets.size() * sizeof(unsigned int)));
  cudaCheck(cudaMalloc((void**)&dev_hit_phi, acc_hits * sizeof(float)));
  cudaCheck(cudaMalloc((void**)&dev_hit_temp, acc_hits * sizeof(int32_t)));
  cudaCheck(cudaMalloc((void**)&dev_hit_permutation, acc_hits * sizeof(unsigned short)));

  // Copy required data
  cudaCheck(cudaMemcpy(dev_event_offsets, event_offsets.data(), event_offsets.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_hit_offsets, hit_offsets.data(), hit_offsets.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
  acc_size = 0;
  for (unsigned int i=0; i<number_of_events; ++i){
    const auto event_no = start_event + i;
    cudaCheck(cudaMemcpy(&dev_input[acc_size], input[event_no].data(), input[event_no].size(), cudaMemcpyHostToDevice));
    acc_size += input[event_no].size();
  }

  // // Invoke kernel
  auto calculatePhiAndSort = CalculatePhiAndSort(
    num_blocks,
    sort_num_threads,
    stream,
    dev_input,
    dev_event_offsets,
    dev_hit_offsets,
    dev_hit_phi,
    dev_hit_temp,
    dev_hit_permutation
  );

  times.emplace_back(
    Helper::invoke(calculatePhiAndSort)
  );

  cudaCheck(cudaPeekAtLastError());

  // Free buffers
  cudaCheck(cudaFree(dev_hit_permutation));

  /////////////////////
  // SearchByTriplet //
  /////////////////////

  // Allocate buffers
  cudaCheck(cudaMalloc((void**)&dev_tracks, number_of_events * MAX_TRACKS * sizeof(Track)));
  cudaCheck(cudaMalloc((void**)&dev_tracks_to_follow, number_of_events * TTF_MODULO * sizeof(unsigned int)));
  cudaCheck(cudaMalloc((void**)&dev_hit_used, acc_hits * sizeof(bool)));
  cudaCheck(cudaMalloc((void**)&dev_atomics_storage, number_of_events * atomic_space * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_tracklets, acc_hits * sizeof(Track)));
  cudaCheck(cudaMalloc((void**)&dev_weak_tracks, acc_hits * sizeof(unsigned int)));
  cudaCheck(cudaMalloc((void**)&dev_h0_candidates, 2 * acc_hits * sizeof(short)));
  cudaCheck(cudaMalloc((void**)&dev_h2_candidates, 2 * acc_hits * sizeof(short)));
  cudaCheck(cudaMalloc((void**)&dev_rel_indices, number_of_events * MAX_NUMHITS_IN_MODULE * sizeof(unsigned short)));
  
  // Initialize data
  cudaCheck(cudaMemset(dev_hit_used, false, acc_hits * sizeof(bool)));
  cudaCheck(cudaMemset(dev_atomics_storage, 0, number_of_events * atomic_space * sizeof(int)));

  // Invoke kernel
  auto searchByTriplet = SearchByTriplet(
    num_blocks,
    sbt_num_threads,
    stream,
    dev_tracks,
    dev_input,
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
    Helper::invoke(searchByTriplet)
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

  ////////////////////////
  // Data consolidation //
  ////////////////////////

  std::vector<unsigned int> track_start (number_of_events);
  std::vector<unsigned int> number_of_tracks (number_of_events);
  cudaCheck(cudaMemcpy(number_of_tracks.data(), dev_atomics_storage, number_of_events * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  
  unsigned int total_number_of_tracks = 0;
  for (unsigned int event_no=0; event_no<number_of_events; ++event_no) {
    track_start[event_no] = total_number_of_tracks;
    total_number_of_tracks += number_of_tracks[event_no];
  }
  
  // DEBUG << "Found a total of " << total_number_of_tracks << " tracks ("
  //   << (total_number_of_tracks / ((float) number_of_tracks.size()))
  //   << " average tracks per event)" << std::endl
  //   << "Max tracks in one event: " << *(std::max_element(number_of_tracks.begin(), number_of_tracks.end())) << std::endl;

  // Reserve exactly the amount of memory we need for consolidated tracks and VeloStates
  auto consolidated_tracks_size = (1 + number_of_events * 2) * sizeof(unsigned int)
                                  + total_number_of_tracks * sizeof(Track);
  cudaCheck(cudaMalloc((void**)&dev_consolidated_tracks, consolidated_tracks_size));

  // Copy tracks data into consolidated tracks
  char* dev_consolidated_tracks_pointer = dev_consolidated_tracks;
  cudaCheck(cudaMemcpy(dev_consolidated_tracks_pointer, (char*) &total_number_of_tracks, sizeof(unsigned int), cudaMemcpyHostToDevice));
  dev_consolidated_tracks_pointer += sizeof(unsigned int);
  cudaCheck(cudaMemcpy(dev_consolidated_tracks_pointer, track_start.data(), number_of_events * sizeof(unsigned int), cudaMemcpyHostToDevice));
  dev_consolidated_tracks_pointer += number_of_events * sizeof(unsigned int);
  cudaCheck(cudaMemcpy(dev_consolidated_tracks_pointer, (char*) dev_atomics_storage, number_of_events * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  dev_consolidated_tracks_pointer += number_of_events * sizeof(unsigned int);
  // Copy tracks for each event
  for (unsigned int event_no=0; event_no<number_of_events; ++event_no) {
    cudaCheck(cudaMemcpy(dev_consolidated_tracks_pointer, (char*) (dev_tracks + event_no * MAX_TRACKS), number_of_tracks[event_no] * sizeof(Track), cudaMemcpyDeviceToDevice));
    dev_consolidated_tracks_pointer += number_of_tracks[event_no] * sizeof(Track);
  }

  // DEBUG << "Original tracks container (B): " << number_of_events * MAX_TRACKS * sizeof(Track) << std::endl
  //   << "Consolidated tracks container (B): " << (dev_consolidated_tracks_pointer - dev_consolidated_tracks)
  //   << " (" << 100.f * ((float) (dev_consolidated_tracks_pointer - dev_consolidated_tracks)) / (number_of_events * MAX_TRACKS * sizeof(Track)) << " %)"
  //   << std::endl;

  // Free buffers
  cudaCheck(cudaFree(dev_atomics_storage));
  cudaCheck(cudaFree(dev_tracks));

  ///////////////////////////
  // Calculate VELO states //
  ///////////////////////////

  // Allocate buffers
  cudaCheck(cudaMalloc((void**)&dev_velo_states, total_number_of_tracks * STATES_PER_TRACK * sizeof(VeloState)));

  // Invoke kernel
  auto calculateVeloStates = CalculateVeloStates(
    num_blocks,
    velo_states_num_threads,
    stream,
    dev_input,
    dev_consolidated_tracks,
    dev_velo_states,
    dev_hit_temp,
    dev_event_offsets,
    dev_hit_offsets
  );

  times.emplace_back(
    Helper::invoke(calculateVeloStates)
  );

  cudaCheck(cudaPeekAtLastError());

  // TODO: The chain could follow from here on.
  // If the chain follows, we may not need to retrieve the data
  // in the state it is currently, but in a posterior state.
  // In principle, here we need to get back:
  // - dev_input: Shuffled input
  // - dev_hit_temp: hit Xs
  // - dev_consolidated_tracks: Consolidated tracks
  // - dev_velo_states: VELO filtered states for each track
  
  // Therefore, this is just temporal
  // Free buffers
  cudaCheck(cudaFree(dev_event_offsets));
  cudaCheck(cudaFree(dev_hit_offsets));

  // Fetch required data
  std::vector<uint8_t> output (acc_size);
  std::vector<float> hit_xs (acc_hits);
  std::vector<char> consolidated_tracks (consolidated_tracks_size);
  std::vector<VeloState> velo_states (total_number_of_tracks * STATES_PER_TRACK);

  cudaCheck(cudaMemcpy(output.data(), dev_input, acc_size, cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(hit_xs.data(), dev_hit_temp, acc_hits * sizeof(float), cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(consolidated_tracks.data(), dev_consolidated_tracks, consolidated_tracks_size, cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(velo_states.data(), dev_velo_states, total_number_of_tracks * STATES_PER_TRACK * sizeof(VeloState), cudaMemcpyDeviceToHost));

  // Free buffers
  cudaCheck(cudaFree(dev_input));
  cudaCheck(cudaFree(dev_hit_temp));
  cudaCheck(cudaFree(dev_consolidated_tracks));
  cudaCheck(cudaFree(dev_velo_states));

  if (do_print_timing) {
    print_timing(number_of_events, times);
  }

  return cudaSuccess;
}

void Stream::print_timing(
  const unsigned int number_of_events,
  const std::vector<float>& times
) {
  DEBUG << std::endl << "Time averages:" << std::endl
    << " Phi + sorting throughput: " << number_of_events / (times[0] * 0.001)
    << " events/s (" << times[0] << " ms)" << std::endl;

  auto accumulated_time = times[0] + times[1];
  DEBUG << " Search by triplet: "
    << number_of_events / (times[1] * 0.001) << " events/s (" << times[1] << " ms), "
    << number_of_events / (accumulated_time * 0.001) << " events/s integrated throughput"
    << std::endl;

  accumulated_time += times[2];
  DEBUG << " Fit: "
    << number_of_events / (times[2] * 0.001) << " fits/s (" << times[2] << " ms), "
    << number_of_events / (accumulated_time * 0.001) << " events/s integrated throughput"
    << std::endl;
}
