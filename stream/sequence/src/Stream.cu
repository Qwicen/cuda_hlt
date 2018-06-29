#include "../include/Stream.cuh"

cudaError_t Stream::operator()(
  const char* host_events_pinned,
  const uint* host_event_offsets_pinned,
  size_t host_events_pinned_size,
  size_t host_event_offsets_pinned_size,
  uint number_of_events,
  uint number_of_repetitions
) {
  for (uint repetition=0; repetition<number_of_repetitions; ++repetition) {
    std::vector<std::pair<std::string, float>> times;
    Timer t_total;

    ////////////////
    // Clustering //
    ////////////////

    if (transmit_host_to_device) {
      cudaCheck(cudaMemcpyAsync(std::get<0>(sequence.item<seq::estimate_input_size>().arguments), host_events_pinned, host_events_pinned_size, cudaMemcpyHostToDevice, stream));
      // cudaCheck(cudaMemcpyAsync(std::get<0>(std::get<0>(sequence.algorithms).arguments), host_events_pinned, host_events_pinned_size, cudaMemcpyHostToDevice, stream));
      cudaCheck(cudaMemcpyAsync(std::get<1>(sequence.item<seq::estimate_input_size>().arguments), host_event_offsets_pinned, host_event_offsets_pinned_size * sizeof(uint), cudaMemcpyHostToDevice, stream));
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

    // Convert the estimated sizes to module hit start format (offsets)
    Helper::invoke(
      sequence.item<seq::prefix_sum_reduce>(),
      "Prefix sum reduce",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    Helper::invoke(
      sequence.item<seq::prefix_sum_single_block>(),
      "Prefix sum single block",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    Helper::invoke(
      sequence.item<seq::prefix_sum_scan>(),
      "Prefix sum scan",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    // // Fetch the number of hits we require
    // uint number_of_hits;
    // cudaCheck(cudaMemcpyAsync(&number_of_hits, estimateInputSize.dev_estimated_input_size + number_of_events * VeloTracking::n_modules, sizeof(uint), cudaMemcpyDeviceToHost, stream));
    // const auto required_size = number_of_hits * 6;

    // if (required_size > velo_cluster_container_size) {
    //   warning_cout << "Number of hits: " << number_of_hits << std::endl
    //     << "Size of velo cluster container is larger than previously accomodated." << std::endl
    //     << "Resizing from " << velo_cluster_container_size * sizeof(uint) << " to " << required_size * sizeof(uint) << " B" << std::endl;

    //   cudaCheck(cudaFree(maskedVeloClustering.dev_velo_cluster_container));
    //   cudaCheck(cudaMalloc((void**)&maskedVeloClustering.dev_velo_cluster_container, required_size * sizeof(uint)));
    // }

    // Invoke clustering
    Helper::invoke(
      sequence.item<seq::masked_velo_clustering>(),
      "Masked velo clustering",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    // Print output
    // maskedVeloClustering.print_output(number_of_events, 3);

    // if (do_check) {
    //   // Check results
    //   maskedVeloClustering.check(
    //     host_events_pinned,
    //     host_event_offsets_pinned,
    //     host_events_pinned_size,
    //     host_event_offsets_pinned_size,
    //     geometry,
    //     number_of_events
    //   );
    // }

    /////////////////////////
    // CalculatePhiAndSort //
    /////////////////////////

    Helper::invoke(
      sequence.item<seq::calculate_phi_and_sort>(),
      "Calculate phi and sort",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    // Print output
    // calculatePhiAndSort.print_output(number_of_events);

    /////////////////////
    // SearchByTriplet //
    /////////////////////

    Helper::invoke(
      sequence.item<seq::search_by_triplet>(),
      "Search by triplet",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    // Print output
    // searchByTriplet.print_output(number_of_events);
    
    ////////////////////////
    // Consolidate tracks //
    ////////////////////////
    
    Helper::invoke(
      sequence.item<seq::copy_and_prefix_sum_single_block>(),
      "Calculate accumulated tracks",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
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
      cudaCheck(cudaMemcpyAsync(host_number_of_tracks_pinned, std::get<8>(sequence.item<seq::search_by_triplet>().arguments), number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_tracks_pinned, std::get<2>(sequence.item<seq::consolidate_tracks>().arguments), number_of_events * max_tracks_in_event * sizeof(Track<mc_check_enabled>), cudaMemcpyDeviceToHost, stream));
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
        cudaCheck(cudaMemcpyAsync(host_number_of_tracks_pinned, std::get<8>(sequence.item<seq::search_by_triplet>().arguments), number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaMemcpyAsync(host_accumulated_tracks, (void*)(std::get<8>(sequence.item<seq::search_by_triplet>().arguments) + number_of_events), number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaMemcpyAsync(host_tracks_pinned, std::get<2>(sequence.item<seq::consolidate_tracks>().arguments), number_of_events * max_tracks_in_event * sizeof(Track<mc_check_enabled>), cudaMemcpyDeviceToHost, stream));

        cudaEventRecord(cuda_generic_event, stream);
        cudaEventSynchronize(cuda_generic_event);

        checkTracks(reinterpret_cast<Track<true>*>(host_tracks_pinned),
          host_accumulated_tracks,
  		    host_number_of_tracks_pinned,
          number_of_events,
  		    folder_name_MC);
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
