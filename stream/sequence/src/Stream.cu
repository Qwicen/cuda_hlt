#include "../include/Stream.cuh"
#include "../../../main/include/Common.h"
#include "../include/run_VeloUT_CPU.h"

cudaError_t Stream::operator()(
  const char* host_events_pinned,
  const uint* host_event_offsets_pinned,
  size_t host_events_pinned_size,
  size_t host_event_offsets_pinned_size,
  VeloUTTracking::HitsSoA *hits_layers_events_ut,
  const uint32_t n_hits_layers_events_ut[][VeloUTTracking::n_layers],
  uint number_of_events,
  uint number_of_repetitions,
  uint i_stream
) {
  for (uint repetition=0; repetition<number_of_repetitions; ++repetition) {
    std::vector<std::pair<std::string, float>> times;
    Timer t_total;

    ////////////////
    // Clustering //
    ////////////////

    if (transmit_host_to_device) {
      cudaCheck(cudaMemcpyAsync(estimateInputSize.dev_raw_input, host_events_pinned, host_events_pinned_size, cudaMemcpyHostToDevice, stream));
      cudaCheck(cudaMemcpyAsync(estimateInputSize.dev_raw_input_offsets, host_event_offsets_pinned, host_event_offsets_pinned_size * sizeof(uint), cudaMemcpyHostToDevice, stream));
    }

    // Estimate the input size of each module
    Helper::invoke(
      estimateInputSize,
      "Estimate input size",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    // Convert the estimated sizes to module hit start format (offsets)
    Helper::invoke(
      prefixSumReduce,
      "Prefix sum reduce",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
     );

    Helper::invoke(
      prefixSumSingleBlock,
      "Prefix sum single block",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    Helper::invoke(
      prefixSumScan,
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
      maskedVeloClustering,
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
      calculatePhiAndSort,
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
      searchByTriplet,
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
      copyAndPrefixSumSingleBlock,
      "Calculate accumulated tracks",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
     );

    Helper::invoke(
      consolidateTracks,
      "Consolidate tracks",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );

    ////////////////////////////////////////
    // Optional: Simplified Kalman filter //
    ////////////////////////////////////////

    if (do_simplified_kalman_filter) {
      Helper::invoke(
        simplifiedKalmanFilter,
        "Simplified Kalman filter",
        times,
        cuda_event_start,
        cuda_event_stop,
        print_individual_rates
      );
    }
    
    // Transmission device to host
    if (transmit_device_to_host) {
      cudaCheck(cudaMemcpyAsync(host_number_of_tracks_pinned, searchByTriplet.dev_atomics_storage, number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_tracks_pinned, consolidateTracks.dev_output_tracks, number_of_events * max_tracks_in_event * sizeof(VeloTracking::Track<mc_check_enabled>), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_accumulated_tracks, (void*)(searchByTriplet.dev_atomics_storage + number_of_events), number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
      if ( do_simplified_kalman_filter ) {
	cudaCheck(cudaMemcpyAsync(host_velo_states, consolidateTracks.dev_velo_states, number_of_events * max_tracks_in_event * VeloTracking::states_per_track * sizeof(VeloState), cudaMemcpyDeviceToHost, stream));
      }
      else {
	cudaCheck(cudaMemcpyAsync(host_velo_states, consolidateTracks.dev_velo_states, number_of_events * max_tracks_in_event * sizeof(VeloState), cudaMemcpyDeviceToHost, stream));
      }
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


    if (do_check && i_stream == 0) {
      if (repetition == 0) { // only check efficiencies once
        // Fetch data
        cudaCheck(cudaMemcpyAsync(host_number_of_tracks_pinned, searchByTriplet.dev_atomics_storage, number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaMemcpyAsync(host_accumulated_tracks, (void*)(searchByTriplet.dev_atomics_storage + number_of_events), number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaMemcpyAsync(host_tracks_pinned, consolidateTracks.dev_output_tracks, number_of_events * max_tracks_in_event * sizeof(VeloTracking::Track<mc_check_enabled>), cudaMemcpyDeviceToHost, stream));
        cudaEventRecord(cuda_generic_event, stream);
        cudaEventSynchronize(cuda_generic_event);

	std::cout << "CHECKING VELO TRACKS " << std::endl;
	
        const std::vector< trackChecker::Tracks > tracks_events = prepareTracks(
          host_tracks_pinned,
      	  host_accumulated_tracks,
      	  host_number_of_tracks_pinned,
      	  number_of_events);
      
        const bool fromNtuple = true;
        const std::string trackType = "Velo";
      	call_pr_checker (
	  tracks_events,
      	  folder_name_MC,
    	  fromNtuple,
    	  trackType);
      }
    }

    /* Plugin VeloUT CPU code here 
       Adjust input types to match PrVeloUT code
    */
    if (mc_check_enabled && i_stream == 0) {

      std::vector< trackChecker::Tracks > *ut_tracks_events = new std::vector< trackChecker::Tracks >;
      
      int rv = run_veloUT_on_CPU(
	         ut_tracks_events,
		 hits_layers_events_ut,
		 n_hits_layers_events_ut,
		 host_velo_states,
		 host_accumulated_tracks,
		 host_tracks_pinned,
		 host_number_of_tracks_pinned,
		 number_of_events
	       );

      if ( rv != 0 )
	continue;
      
      
      std::cout << "CHECKING VeloUT TRACKS" << std::endl;
      const bool fromNtuple = true;
      const std::string trackType = "VeloUT";
      call_pr_checker (
        *ut_tracks_events,
	folder_name_MC,
	fromNtuple,
	trackType); 
      
      delete ut_tracks_events;
      
      
    } // mc_check_enabled      
    
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
