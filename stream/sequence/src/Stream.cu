#include "../include/Stream.cuh"

cudaError_t Stream::operator()(
  const char* host_events,
  const uint* host_event_offsets,
  size_t host_events_size,
  size_t host_event_offsets_size,
  uint number_of_events,
  uint number_of_repetitions
) {
  const bool do_print_memory_manager = false;

  // Generate object for populating arguments
  DynamicArgumentGenerator<decltype(arguments)> argen {arguments, dev_base_pointer};

  for (uint repetition=0; repetition<number_of_repetitions; ++repetition) {
    std::vector<std::pair<std::string, float>> times;
    Timer t_total;
    std::map<uint, uint> offsets;
    uint sequence_step = 0;

    // Reset scheduler
    scheduler.reset();

    ////////////////
    // Clustering //
    ////////////////

    // Reserve memory for this step datatypes
    scheduler.setup_next(
      std::map<uint, size_t>{
        argen.size_pair<arg::dev_raw_input>(host_events_size),
        argen.size_pair<arg::dev_raw_input_offsets>(host_event_offsets_size),
        argen.size_pair<arg::dev_estimated_input_size>(number_of_events * VeloTracking::n_modules + 1),
        argen.size_pair<arg::dev_module_cluster_num>(number_of_events * VeloTracking::n_modules),
        argen.size_pair<arg::dev_module_candidate_num>(number_of_events),
        argen.size_pair<arg::dev_cluster_candidates>(number_of_events * VeloClustering::max_candidates_event)
      },
      offsets,
      sequence_step++,
      do_print_memory_manager
    );

    // Setup arguments for kernel call
    sequence.item<seq::estimate_input_size>().set_arguments(
      argen.generate<arg::dev_raw_input>(offsets),
      argen.generate<arg::dev_raw_input_offsets>(offsets),
      argen.generate<arg::dev_estimated_input_size>(offsets),
      argen.generate<arg::dev_module_cluster_num>(offsets),
      argen.generate<arg::dev_module_candidate_num>(offsets),
      argen.generate<arg::dev_cluster_candidates>(offsets)
    );

    if (transmit_host_to_device) {
      cudaCheck(cudaMemcpyAsync(argen.generate<arg::dev_raw_input>(offsets), host_events, host_events_size, cudaMemcpyHostToDevice, stream));
      cudaCheck(cudaMemcpyAsync(argen.generate<arg::dev_raw_input_offsets>(offsets), host_event_offsets, host_event_offsets_size * sizeof(uint), cudaMemcpyHostToDevice, stream));
      cudaEventRecord(cuda_generic_event, stream);
      cudaEventSynchronize(cuda_generic_event);
    }

    // Estimate the input size of each module
    Helper::invoke(
      sequence.item<seq::estimate_input_size>(),
      "Estimate input size",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    // Reserve memory
    scheduler.setup_next(
      {
        argen.size_pair<arg::dev_cluster_offset>(number_of_events)
      },
      offsets,
      sequence_step++,
      do_print_memory_manager
    );

    // Setup sequence step
    sequence.item<seq::prefix_sum_reduce>().set_arguments(
      argen.generate<arg::dev_estimated_input_size>(offsets),
      argen.generate<arg::dev_cluster_offset>(offsets),
      VeloTracking::n_modules * number_of_events
    );

    // Convert the estimated sizes to module hit start format (offsets)
    Helper::invoke(
      sequence.item<seq::prefix_sum_reduce>(),
      "Prefix sum reduce",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    // Reserve memory
    scheduler.setup_next(
      {},
      offsets,
      sequence_step++,
      do_print_memory_manager
    );

    // Setup sequence step
    sequence.item<seq::prefix_sum_single_block>().set_arguments(
      argen.generate<arg::dev_estimated_input_size>(offsets) + VeloTracking::n_modules * number_of_events,
      argen.generate<arg::dev_cluster_offset>(offsets),
      prefixSumBlocks
    );

    Helper::invoke(
      sequence.item<seq::prefix_sum_single_block>(),
      "Prefix sum single block",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    // Reserve memory
    scheduler.setup_next(
      {},
      offsets,
      sequence_step++,
      do_print_memory_manager
    );

    // Setup sequence step
    sequence.item<seq::prefix_sum_scan>().set_arguments(
      argen.generate<arg::dev_estimated_input_size>(offsets),
      argen.generate<arg::dev_cluster_offset>(offsets),
      VeloTracking::n_modules * number_of_events
    );

    Helper::invoke(
      sequence.item<seq::prefix_sum_scan>(),
      "Prefix sum scan",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    // Fetch the number of hits we require
    cudaCheck(cudaMemcpyAsync(host_total_number_of_velo_clusters, argen.generate<arg::dev_estimated_input_size>(offsets) + number_of_events * VeloTracking::n_modules, sizeof(uint), cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    // Reserve memory
    scheduler.setup_next(
      {
        argen.size_pair<arg::dev_velo_cluster_container>(6 * host_total_number_of_velo_clusters[0])
      },
      offsets,
      sequence_step++,
      do_print_memory_manager
    );

    // Setup sequence step
    sequence.item<seq::masked_velo_clustering>().set_arguments(
      argen.generate<arg::dev_raw_input>(offsets),
      argen.generate<arg::dev_raw_input_offsets>(offsets),
      argen.generate<arg::dev_estimated_input_size>(offsets),
      argen.generate<arg::dev_module_cluster_num>(offsets),
      argen.generate<arg::dev_module_candidate_num>(offsets),
      argen.generate<arg::dev_cluster_candidates>(offsets),
      argen.generate<arg::dev_velo_cluster_container>(offsets),
      dev_velo_geometry
    );

    // Invoke clustering
    Helper::invoke(
      sequence.item<seq::masked_velo_clustering>(),
      "Masked velo clustering",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    /////////////////////////
    // CalculatePhiAndSort //
    /////////////////////////

    // Reserve memory
    scheduler.setup_next(
      {
        argen.size_pair<arg::dev_hit_permutation>(host_total_number_of_velo_clusters[0])
      },
      offsets,
      sequence_step++,
      do_print_memory_manager
    );

    // Setup sequence step
    sequence.item<seq::calculate_phi_and_sort>().set_arguments(
      argen.generate<arg::dev_estimated_input_size>(offsets),
      argen.generate<arg::dev_module_cluster_num>(offsets),
      argen.generate<arg::dev_velo_cluster_container>(offsets),
      argen.generate<arg::dev_hit_permutation>(offsets)
    );

    Helper::invoke(
      sequence.item<seq::calculate_phi_and_sort>(),
      "Calculate phi and sort",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    /////////////////////
    // SearchByTriplet //
    /////////////////////

    // Reserve memory
    scheduler.setup_next(
      {
        argen.size_pair<arg::dev_tracks>(number_of_events * VeloTracking::max_tracks),
        argen.size_pair<arg::dev_tracklets>(number_of_events * VeloTracking::ttf_modulo),
        argen.size_pair<arg::dev_tracks_to_follow>(number_of_events * VeloTracking::ttf_modulo),
        argen.size_pair<arg::dev_weak_tracks>(number_of_events * VeloTracking::ttf_modulo),
        argen.size_pair<arg::dev_hit_used>(host_total_number_of_velo_clusters[0]),
        argen.size_pair<arg::dev_atomics_storage>(number_of_events * VeloTracking::num_atomics),
        argen.size_pair<arg::dev_h0_candidates>(2 * host_total_number_of_velo_clusters[0]),
        argen.size_pair<arg::dev_h2_candidates>(2 * host_total_number_of_velo_clusters[0]),
        argen.size_pair<arg::dev_rel_indices>(number_of_events * VeloTracking::max_numhits_in_module)
      },
      offsets,
      sequence_step++,
      do_print_memory_manager
    );

    // Setup sequence step
    sequence.item<seq::search_by_triplet>().set_arguments(
      argen.generate<arg::dev_velo_cluster_container>(offsets),
      argen.generate<arg::dev_estimated_input_size>(offsets),
      argen.generate<arg::dev_module_cluster_num>(offsets),
      argen.generate<arg::dev_tracks>(offsets),
      argen.generate<arg::dev_tracklets>(offsets),
      argen.generate<arg::dev_tracks_to_follow>(offsets),
      argen.generate<arg::dev_weak_tracks>(offsets),
      argen.generate<arg::dev_hit_used>(offsets),
      argen.generate<arg::dev_atomics_storage>(offsets),
      argen.generate<arg::dev_h0_candidates>(offsets),
      argen.generate<arg::dev_h2_candidates>(offsets),
      argen.generate<arg::dev_rel_indices>(offsets)
    );

    Helper::invoke(
      sequence.item<seq::search_by_triplet>(),
      "Search by triplet",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );
    
    ////////////////////////
    // Consolidate tracks //
    ////////////////////////

    // Calculate accumulated tracks

    // Reserve memory
    scheduler.setup_next(
      {},
      offsets,
      sequence_step++,
      do_print_memory_manager
    );

    // Setup sequence step
    sequence.item<seq::copy_and_prefix_sum_single_block>().set_arguments(
      (uint*) argen.generate<arg::dev_atomics_storage>(offsets) + number_of_events*2,
      (uint*) argen.generate<arg::dev_atomics_storage>(offsets),
      (uint*) argen.generate<arg::dev_atomics_storage>(offsets) + number_of_events,
      number_of_events
    );
    
    Helper::invoke(
      sequence.item<seq::copy_and_prefix_sum_single_block>(),
      "Calculate accumulated tracks",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    // Fetch number of reconstructed tracks
    cudaCheck(cudaMemcpyAsync(host_number_of_reconstructed_velo_tracks, argen.generate<arg::dev_atomics_storage>(offsets) + number_of_events * 2, sizeof(uint), cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);
    size_t velo_track_hit_number_size = host_number_of_reconstructed_velo_tracks[0] + 1;

    // Prefix sum of accumulated tracks
    // 1. Copy velo track hit number to a consecutive container
    // 2. Reduce
    // 3. Single block
    // 4. Scan

    // Reserve memory
    scheduler.setup_next(
      {
        argen.size_pair<arg::dev_velo_track_hit_number>(velo_track_hit_number_size)
      },
      offsets,
      sequence_step++,
      do_print_memory_manager
    );

    // Setup sequence step
    sequence.item<seq::copy_velo_track_hit_number>().set_arguments(
      argen.generate<arg::dev_tracks>(offsets),
      argen.generate<arg::dev_atomics_storage>(offsets),
      argen.generate<arg::dev_velo_track_hit_number>(offsets)
    );

    Helper::invoke(
      sequence.item<seq::copy_velo_track_hit_number>(),
      "Copy Velo track hit number",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    // Prefix sum in three kernels
    const size_t prefix_sum_auxiliary_array_2_size = (host_number_of_reconstructed_velo_tracks[0] + 511) / 512;
    scheduler.setup_next(
      {
        argen.size_pair<arg::dev_prefix_sum_auxiliary_array_2>(prefix_sum_auxiliary_array_2_size)
      },
      offsets,
      sequence_step++,
      do_print_memory_manager
    );

    // Setup sequence step
    sequence.item<seq::prefix_sum_reduce_velo_track_hit_number>().set_arguments(
      argen.generate<arg::dev_velo_track_hit_number>(offsets),
      argen.generate<arg::dev_prefix_sum_auxiliary_array_2>(offsets),
      host_number_of_reconstructed_velo_tracks[0]
    );

    // Setup sequence opts
    sequence.item<seq::prefix_sum_reduce_velo_track_hit_number>().set_opts(dim3(prefix_sum_auxiliary_array_2_size), dim3(256), stream);

    // Convert the estimated sizes to module hit start format (offsets)
    Helper::invoke(
      sequence.item<seq::prefix_sum_reduce_velo_track_hit_number>(),
      "Prefix sum reduce: Velo track hit number",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    // Reserve memory
    scheduler.setup_next({}, offsets, sequence_step++, do_print_memory_manager);

    // Setup sequence step
    sequence.item<seq::prefix_sum_single_block_velo_track_hit_number>().set_arguments(
      argen.generate<arg::dev_velo_track_hit_number>(offsets) + host_number_of_reconstructed_velo_tracks[0],
      argen.generate<arg::dev_prefix_sum_auxiliary_array_2>(offsets),
      prefix_sum_auxiliary_array_2_size
    );

    Helper::invoke(
      sequence.item<seq::prefix_sum_single_block_velo_track_hit_number>(),
      "Prefix sum single block: Velo track hit number",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    // Reserve memory
    scheduler.setup_next({}, offsets, sequence_step++, do_print_memory_manager);

    // Setup sequence step
    sequence.item<seq::prefix_sum_scan_velo_track_hit_number>().set_arguments(
      argen.generate<arg::dev_velo_track_hit_number>(offsets),
      argen.generate<arg::dev_prefix_sum_auxiliary_array_2>(offsets),
      host_number_of_reconstructed_velo_tracks[0]
    );

    // Setup sequence opts
    const uint pss_velo_track_hit_number_opts =
      prefix_sum_auxiliary_array_2_size==1 ? 1 : (prefix_sum_auxiliary_array_2_size-1);
    sequence.item<seq::prefix_sum_scan_velo_track_hit_number>().set_opts(dim3(pss_velo_track_hit_number_opts), dim3(512), stream);

    Helper::invoke(
      sequence.item<seq::prefix_sum_scan_velo_track_hit_number>(),
      "Prefix sum scan",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    // Fetch total number of hits accumulated
    // with all tracks
    cudaCheck(cudaMemcpyAsync(host_accumulated_number_of_hits_in_velo_tracks,
      argen.generate<arg::dev_velo_track_hit_number>(offsets) + host_number_of_reconstructed_velo_tracks[0],
      sizeof(uint), cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    // Reserve memory
    scheduler.setup_next(
      {
        argen.size_pair<arg::dev_velo_track_hits>(host_accumulated_number_of_hits_in_velo_tracks[0]),
        argen.size_pair<arg::dev_velo_states>(host_number_of_reconstructed_velo_tracks[0])
      },
      offsets,
      sequence_step++,
      do_print_memory_manager
    );

    // Setup sequence step
    sequence.item<seq::consolidate_tracks>().set_arguments(
      argen.generate<arg::dev_atomics_storage>(offsets),
      argen.generate<arg::dev_tracks>(offsets),
      argen.generate<arg::dev_velo_track_hit_number>(offsets),
      argen.generate<arg::dev_velo_cluster_container>(offsets),
      argen.generate<arg::dev_estimated_input_size>(offsets),
      argen.generate<arg::dev_module_cluster_num>(offsets),
      argen.generate<arg::dev_velo_track_hits>(offsets),
      argen.generate<arg::dev_velo_states>(offsets)
    );    

    Helper::invoke(
      sequence.item<seq::consolidate_tracks>(),
      "Consolidate tracks",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    ////////////////////////////////////////
    // Optional: Simplified Kalman filter //
    ////////////////////////////////////////

    // if (do_simplified_kalman_filter) {
    //   Helper::invoke(
    //     simplifiedKalmanFilter,
    //     "Simplified Kalman filter",
    //     times,
    //     cuda_event_start,
    //     cuda_event_stop,
    //     print_individual_rates
    //   );
    // }
    
    // Transmission device to host
    if (transmit_device_to_host) {
      cudaCheck(cudaMemcpyAsync(host_number_of_tracks, argen.generate<arg::dev_atomics_storage>(offsets), number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_accumulated_tracks, argen.generate<arg::dev_atomics_storage>(offsets) + number_of_events, number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_velo_track_hit_number, argen.generate<arg::dev_velo_track_hit_number>(offsets), argen.size<arg::dev_velo_track_hit_number>(velo_track_hit_number_size), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_velo_track_hits, argen.generate<arg::dev_velo_track_hits>(offsets), argen.size<arg::dev_velo_track_hits>(host_accumulated_number_of_hits_in_velo_tracks[0]), cudaMemcpyDeviceToHost, stream));
    }

    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    if (print_individual_rates) {
      t_total.stop();
      times.emplace_back("total", t_total.get());
      print_timing(number_of_events, times);
    }

    ///////////////////////
    // Monte Carlo Check //
    ///////////////////////

    if (mc_check_enabled) {
      if (repetition == 0 && do_check) { // only check efficiencies once
        // Fetch data
        cudaCheck(cudaMemcpyAsync(host_number_of_tracks, argen.generate<arg::dev_atomics_storage>(offsets), number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaMemcpyAsync(host_accumulated_tracks, argen.generate<arg::dev_atomics_storage>(offsets) + number_of_events, number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaMemcpyAsync(host_velo_track_hit_number, argen.generate<arg::dev_velo_track_hit_number>(offsets), argen.size<arg::dev_velo_track_hit_number>(velo_track_hit_number_size), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaMemcpyAsync(host_velo_track_hits, argen.generate<arg::dev_velo_track_hits>(offsets), argen.size<arg::dev_velo_track_hits>(host_accumulated_number_of_hits_in_velo_tracks[0]), cudaMemcpyDeviceToHost, stream));
        cudaEventRecord(cuda_generic_event, stream);
        cudaEventSynchronize(cuda_generic_event);

        checkTracks(
          host_number_of_tracks,
          host_accumulated_tracks,
          host_velo_track_hit_number,
          reinterpret_cast<Hit<true>*>(host_velo_track_hits),
          number_of_events,
          folder_name_MC
        );
      }
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

  info_cout << "stream #" << stream_number << ": "
    << number_of_events / total_time.second << " events/s"
    << ", partial timers (s): " << partial_times
    << std::endl;
}
