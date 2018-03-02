#include "../include/KernelInvoker.cuh"

cudaError_t invokeParallelSearch(
  const std::vector<std::vector<uint8_t>>& input,
  std::vector<std::vector<uint8_t>>& output
) {
  unsigned int eventsToProcess = input.size();
  cudaEvent_t event_start, event_stop;
  cudaEventCreate(&event_start);
  cudaEventCreate(&event_stop);

  // Choose which GPU to run on
  const int device_number = 0;
  cudaCheck(cudaSetDevice(device_number));
  cudaCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
  cudaDeviceProp* device_properties = (cudaDeviceProp*) malloc(sizeof(cudaDeviceProp));
  cudaGetDeviceProperties(device_properties, 0);

  // Blocks and threads
  dim3 numBlocks(eventsToProcess);
  dim3 numThreads(NUMTHREADS_X);

  // Allocate memory
  // Prepare event offset and hit offset
  std::vector<unsigned int> event_offsets;
  std::vector<unsigned int> hit_offsets;
  int acc_size = 0, acc_hits = 0;
  for (unsigned int i=0; i<eventsToProcess; ++i) {
    auto info = EventInfo(input[i]);
    const int event_size = input[i].size();
    event_offsets.push_back(acc_size);
    hit_offsets.push_back(acc_hits);
    acc_size += event_size;
    acc_hits += info.numberOfHits;
  }

  // Number of defined atomics
  constexpr unsigned int atomic_space = NUM_ATOMICS + 1;

  // GPU datatypes
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
  VeloState* dev_velo_states;

  // Allocate GPU buffers
  cudaCheck(cudaMalloc((void**)&dev_tracks, eventsToProcess * MAX_TRACKS * sizeof(Track)));
  cudaCheck(cudaMalloc((void**)&dev_input, acc_size));
  cudaCheck(cudaMalloc((void**)&dev_tracks_to_follow, eventsToProcess * TTF_MODULO * sizeof(unsigned int)));
  cudaCheck(cudaMalloc((void**)&dev_hit_used, acc_hits * sizeof(bool)));
  cudaCheck(cudaMalloc((void**)&dev_atomics_storage, eventsToProcess * atomic_space * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_tracklets, acc_hits * sizeof(Track)));
  cudaCheck(cudaMalloc((void**)&dev_weak_tracks, acc_hits * sizeof(unsigned int)));
  cudaCheck(cudaMalloc((void**)&dev_event_offsets, event_offsets.size() * sizeof(unsigned int)));
  cudaCheck(cudaMalloc((void**)&dev_hit_offsets, hit_offsets.size() * sizeof(unsigned int)));
  cudaCheck(cudaMalloc((void**)&dev_h0_candidates, 2 * acc_hits * sizeof(short)));
  cudaCheck(cudaMalloc((void**)&dev_h2_candidates, 2 * acc_hits * sizeof(short)));
  cudaCheck(cudaMalloc((void**)&dev_rel_indices, eventsToProcess * MAX_NUMHITS_IN_MODULE * sizeof(unsigned short)));
  cudaCheck(cudaMalloc((void**)&dev_hit_phi, acc_hits * sizeof(float)));
  cudaCheck(cudaMalloc((void**)&dev_hit_temp, acc_hits * sizeof(int32_t)));
  cudaCheck(cudaMalloc((void**)&dev_hit_permutation, acc_hits * sizeof(unsigned short)));

  // Copy stuff from host memory to GPU buffers
  cudaCheck(cudaMemcpy(dev_event_offsets, event_offsets.data(), event_offsets.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_hit_offsets, hit_offsets.data(), hit_offsets.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
  acc_size = 0;
  for (unsigned int event_no=0; event_no<eventsToProcess; ++event_no){
    cudaCheck(cudaMemcpy(&dev_input[acc_size], input[event_no].data(), input[event_no].size(), cudaMemcpyHostToDevice));
    acc_size += input[event_no].size();
  }

  // Sorting
  float tsort;
  cudaEventRecord(event_start, 0);

  calculatePhiAndSort<<<numBlocks, 64>>>(
    dev_input,
    dev_event_offsets,
    dev_hit_offsets,
    dev_hit_phi,
    dev_hit_temp,
    dev_hit_permutation
  );

  cudaEventRecord(event_stop, 0);
  cudaEventSynchronize(event_stop);
  cudaEventElapsedTime(&tsort, event_start, event_stop);
  cudaCheck(cudaPeekAtLastError());

  // Repeat the processing several times to average time
  unsigned int niterations = 1;
  unsigned int nexperiments = 1;
  std::vector<std::vector<float>> times_sbt (nexperiments);
  std::vector<std::vector<float>> times_fit (nexperiments);
  std::vector<std::map<std::string, float>> mresults;

  DEBUG << "Now, on your " << device_properties->name
    << ": searchByTriplet with " << eventsToProcess
    << " event" << (eventsToProcess>1 ? "s" : "") << std::endl 
	  << " " << nexperiments << " experiments, "
    << niterations << " iterations" << std::endl;

  for (auto i=0; i<nexperiments; ++i) {

    DEBUG << numThreads.x << ": " << std::flush;

    for (auto j=0; j<niterations; ++j) {
      // Initialize just what we need
      cudaCheck(cudaMemset(dev_hit_used, false, acc_hits * sizeof(bool)));
      cudaCheck(cudaMemset(dev_atomics_storage, 0, eventsToProcess * atomic_space * sizeof(int)));
      
      // searchByTriplet
      float tsbt, tfit;
      cudaEventRecord(event_start, 0);

      searchByTriplet<<<numBlocks, numThreads>>>(
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

      cudaEventRecord(event_stop, 0);
      cudaEventSynchronize(event_stop);
      cudaEventElapsedTime(&tsbt, event_start, event_stop);
      cudaCheck(cudaPeekAtLastError());

      // Consolidate the data for upcoming algorithms
      // Fetch number of tracks for all events
      std::vector<unsigned int> track_start (eventsToProcess);
      std::vector<unsigned int> number_of_tracks (eventsToProcess);
      cudaCheck(cudaMemcpy(number_of_tracks.data(), dev_atomics_storage, eventsToProcess * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      
      unsigned int total_number_of_tracks = 0;
      for (unsigned int event_no=0; event_no<eventsToProcess; ++event_no) {
        track_start[event_no] = total_number_of_tracks;
        total_number_of_tracks += number_of_tracks[event_no];
      }
      
      DEBUG << "Found a total of " << total_number_of_tracks << " tracks ("
        << (total_number_of_tracks / ((float) number_of_tracks.size()))
        << " average tracks per event)" << std::endl
        << "Max tracks in one event: " << *(std::max_element(number_of_tracks.begin(), number_of_tracks.end())) << std::endl;

      // Reserve exactly the amount of memory we need for consolidated tracks and VeloStates
      char* dev_consolidated_tracks;
      VeloState* dev_velo_states;
      auto consolidated_tracks_size = (1 + eventsToProcess * 2) * sizeof(unsigned int)
                                      + total_number_of_tracks * sizeof(Track);
      cudaCheck(cudaMalloc((void**)&dev_consolidated_tracks, consolidated_tracks_size));

      // Copy tracks data into consolidated tracks
      char* dev_consolidated_tracks_pointer = dev_consolidated_tracks;
      cudaCheck(cudaMemcpy(dev_consolidated_tracks_pointer, (char*) &total_number_of_tracks, sizeof(unsigned int), cudaMemcpyHostToDevice));
      dev_consolidated_tracks_pointer += sizeof(unsigned int);

      cudaCheck(cudaMemcpy(dev_consolidated_tracks_pointer, track_start.data(), eventsToProcess * sizeof(unsigned int), cudaMemcpyHostToDevice));
      dev_consolidated_tracks_pointer += eventsToProcess * sizeof(unsigned int);

      cudaCheck(cudaMemcpy(dev_consolidated_tracks_pointer, (char*) dev_atomics_storage, eventsToProcess * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
      dev_consolidated_tracks_pointer += eventsToProcess * sizeof(unsigned int);

      // Copy tracks for each event
      for (unsigned int event_no=0; event_no<eventsToProcess; ++event_no) {
        cudaCheck(cudaMemcpy(dev_consolidated_tracks_pointer, (char*) (dev_tracks + event_no * MAX_TRACKS), number_of_tracks[event_no] * sizeof(Track), cudaMemcpyDeviceToDevice));
        dev_consolidated_tracks_pointer += number_of_tracks[event_no] * sizeof(Track);
      }

      // VeloState* dev_velo_states;
      cudaCheck(cudaMalloc((void**)&dev_velo_states, total_number_of_tracks * STATES_PER_TRACK * sizeof(VeloState)));

      DEBUG << "Original tracks container (B): " << eventsToProcess * MAX_TRACKS * sizeof(Track) << std::endl
        << "Consolidated tracks container (B): " << (dev_consolidated_tracks_pointer - dev_consolidated_tracks)
        << " (" << 100.f * ((float) (dev_consolidated_tracks_pointer - dev_consolidated_tracks)) / (eventsToProcess * MAX_TRACKS * sizeof(Track)) << " %)"
        << std::endl;
      
      // TODO: Revisit free-able objects at each point in the pipeline
      // cudaCheck(cudaFree(dev_atomics_storage));
      // cudaCheck(cudaFree(dev_tracks));

      cudaEventRecord(event_start, 0);

      // Fits
      velo_fit<<<numBlocks, 1024>>>(
        dev_input,
        dev_consolidated_tracks,
        dev_velo_states,
        dev_hit_temp,
        dev_event_offsets,
        dev_hit_offsets
      );

      cudaEventRecord(event_stop, 0);
      cudaEventSynchronize(event_stop);
      cudaEventElapsedTime(&tfit, event_start, event_stop);
      cudaCheck(cudaPeekAtLastError());

      times_sbt[i].push_back(tsbt);
      times_fit[i].push_back(tfit);

      DEBUG << "." << std::flush;
    }

    DEBUG << std::endl;
  }

  if (PRINT_VELO_FIT) {
    std::vector<VeloState> velo_states (eventsToProcess * MAX_TRACKS * STATES_PER_TRACK);
    cudaCheck(cudaMemcpy(velo_states.data(), dev_velo_states, eventsToProcess * MAX_TRACKS * STATES_PER_TRACK * sizeof(VeloState), cudaMemcpyDeviceToHost));

    // Print just first track fits
    for (int i=0; i<3; ++i) {
      const auto& state = velo_states[i];

      std::cout << "At z: " << state.z << std::endl
        << " state (x, y, tx, ty): " << state.x << ", " << state.y << ", " << state.tx << ", " << state.ty << std::endl
        << " covariance (c00, c20, c22, c11, c31, c33): " << state.c00 << ", " << state.c20 << ", " << state.c22 << ", "
        << state.c11 << ", " << state.c31 << ", " << state.c33 << std::endl
        << " chi2: " << state.chi2 << std::endl << std::endl;
    }
  }

  if (PRINT_FILL_CANDIDATES) {
    std::vector<short> h0_candidates (2 * acc_hits);
    std::vector<short> h2_candidates (2 * acc_hits);
    cudaCheck(cudaMemcpy(h0_candidates.data(), dev_h0_candidates, 2 * acc_hits * sizeof(short), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h2_candidates.data(), dev_h2_candidates, 2 * acc_hits * sizeof(short), cudaMemcpyDeviceToHost));
    
    // Just print modules 49, 47 and 45
    auto info = EventInfo(input[0]);

    std::vector<unsigned int> modules {49, 47, 45};
    for (auto module : modules) {
      std::cout << "Module " << module << std::endl << " h0 candidates: ";
      for (auto i=info.module_hitStarts[module]; i<info.module_hitStarts[module]+info.module_hitNums[module]; ++i) {
        std::cout << "(" << h0_candidates[2*i] << ", " << h0_candidates[2*i+1] << ") ";
      }
      std::cout << std::endl;
    }
    
    for (auto module : modules) {
      std::cout << "Module " << module << std::endl << " h2 candidates: ";
      for (auto i=info.module_hitStarts[module]; i<info.module_hitStarts[module]+info.module_hitNums[module]; ++i) {
        std::cout << "(" << h2_candidates[2*i] << ", " << h2_candidates[2*i+1] << ") ";
      }
      std::cout << std::endl;
    }
  }

  // Get results
  if (PRINT_SOLUTION) DEBUG << "Number of tracks found per event:" << std::endl << " ";
  std::vector<int> atomics (eventsToProcess * atomic_space);
  cudaCheck(cudaMemcpy(atomics.data(), dev_atomics_storage, eventsToProcess * atomic_space * sizeof(int), cudaMemcpyDeviceToHost));
  for (unsigned int i=0; i<eventsToProcess; ++i){
    const unsigned int numberOfTracks = atomics[i];
    if (PRINT_SOLUTION) DEBUG << numberOfTracks << ", ";

    std::vector<uint8_t> output_track (numberOfTracks * sizeof(Track));
    cudaCheck(cudaMemcpy(output_track.data(), &dev_tracks[i * MAX_TRACKS], numberOfTracks * sizeof(Track), cudaMemcpyDeviceToHost));
    output.push_back(output_track);
  }
  if (PRINT_SOLUTION) DEBUG << std::endl;

  if (PRINT_VERBOSE) {
    // Print solution of all events processed, to results
    for (unsigned int i=0; i<eventsToProcess; ++i) {

      // Print to output file with event no.
      const int numberOfTracks = output[i].size() / sizeof(Track);
      Track* tracks_in_solution = (Track*) &(output[i])[0];
      std::ofstream outfile (std::string(RESULTS_FOLDER) + std::string("/") + std::to_string(i) + std::string(".txt"));
      for(int j=0; j<numberOfTracks; ++j){
        printTrack(EventInfo(input[i]), tracks_in_solution, j, outfile);
      }
      outfile.close();
    }
  }

  if (PRINT_BINARY) {
    std::cout << "Printing binary solution" << std::endl;
    for (unsigned int i=0; i<eventsToProcess; ++i) {
      const int numberOfTracks = output[i].size() / sizeof(Track);
      Track* tracks_in_solution = (Track*) &(output[i])[0];

      std::ofstream outfile (std::string(RESULTS_FOLDER) + std::string("/") + std::to_string(i) + std::string(".bin"), std::ios::binary);
      outfile.write((char*) &numberOfTracks, sizeof(int32_t));
      // Fetch back the event
      std::vector<uint8_t> event_data (input[i].size());
      cudaCheck(cudaMemcpy(event_data.data(), &dev_input[event_offsets[i]], event_data.size(), cudaMemcpyDeviceToHost));
      auto info = EventInfo(event_data);
      for(int j=0; j<numberOfTracks; ++j){
        writeBinaryTrack((unsigned int*) info.hit_Zs, tracks_in_solution[j], outfile);
      }
      outfile.close();

      if ((i%100) == 0) {
        std::cout << "." << std::flush;
      }
    }
    std::cout << std::endl;
  }

  DEBUG << std::endl << "Time averages:" << std::endl
    << " Phi + sorting throughput: " << eventsToProcess / (tsort * 0.001)
    << " events/s (" << tsort << " ms)" << std::endl;

  int exp = 1;
  for (auto i=0; i<nexperiments; ++i){
    const auto sbt_statistics = calcResults(times_sbt[i]);
    auto accumulated_time = sbt_statistics.at("mean") + tsort;
    
    DEBUG << " Search by triplet (" << NUMTHREADS_X << "): "
      << eventsToProcess / (sbt_statistics.at("mean") * 0.001) << " events/s ("
      << sbt_statistics.at("mean") << " ms, std dev " << sbt_statistics.at("deviation") << "), "
      << eventsToProcess / (accumulated_time * 0.001) << " events/s integrated throughput"
      << std::endl;

    const auto fit_statistics = calcResults(times_fit[i]);
    accumulated_time += fit_statistics.at("mean");
    DEBUG << " Fit: "
      << eventsToProcess / (fit_statistics.at("mean") * 0.001) << " fits/s ("
      << fit_statistics.at("mean") << " ms, std dev " << fit_statistics.at("deviation") << "), "
      << eventsToProcess / (accumulated_time * 0.001) << " events/s integrated throughput"
      << std::endl;

    exp *= 2;
  }
  
  cudaEventDestroy(event_start);
  cudaEventDestroy(event_stop);

  return cudaSuccess;
}
