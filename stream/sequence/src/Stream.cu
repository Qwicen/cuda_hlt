#include "../include/Stream.cuh"
#include "../../../main/include/Common.h"

#include "../../../PrVeloUT/src/PrVeloUT.h"

cudaError_t Stream::operator()(
  const char* host_events_pinned,
  const uint* host_event_offsets_pinned,
  size_t host_events_pinned_size,
  size_t host_event_offsets_pinned_size,
  const VeloUTTracking::HitsSoA hits_layers_events[],
  const uint32_t n_hits_layers_events[][VeloUTTracking::n_layers],
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

    if (do_check) {
      // Check results
      maskedVeloClustering.check(
        host_events_pinned,
        host_event_offsets_pinned,
        host_events_pinned_size,
        host_event_offsets_pinned_size,
        geometry,
        number_of_events
      );
    }

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

    //////////////////////////////
    // Simplified Kalman filter //
    //////////////////////////////

    Helper::invoke(
      simplifiedKalmanFilter,
      "Simplified Kalman filter",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
     );
  
    ////////////////////////
    // Consolidate tracks //
    ////////////////////////
    
    Helper::invoke(
      consolidateTracks,
      "Consolidate tracks",
      times,
      cuda_event_start,
      cuda_event_stop,
      print_individual_rates
    );
    
    // Transmission device to host
    if (transmit_device_to_host) {
      cudaCheck(cudaMemcpyAsync(host_number_of_tracks_pinned, searchByTriplet.dev_atomics_storage, number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_tracks_pinned, consolidateTracks.dev_output_tracks, number_of_events * max_tracks_in_event * sizeof(VeloTracking::Track<do_mc_check>), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_accumulated_tracks, (void*)(searchByTriplet.dev_atomics_storage + number_of_events), number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_velo_states, consolidateTracks.dev_velo_states_out, number_of_events * max_tracks_in_event * VeloTracking::states_per_track * sizeof(VeloState), cudaMemcpyDeviceToHost, stream));
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


    if (do_mc_check) {
      if (repetition == 0) { // only check efficiencies once
        // Fetch data
        cudaCheck(cudaMemcpyAsync(host_number_of_tracks_pinned, searchByTriplet.dev_atomics_storage, number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaMemcpyAsync(host_accumulated_tracks, (void*)(searchByTriplet.dev_atomics_storage + number_of_events), number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaMemcpyAsync(host_tracks_pinned, consolidateTracks.dev_output_tracks, number_of_events * max_tracks_in_event * sizeof(VeloTracking::Track<do_mc_check>), cudaMemcpyDeviceToHost, stream));
        cudaEventRecord(cuda_generic_event, stream);
        cudaEventSynchronize(cuda_generic_event);

        const std::vector< trackChecker::Tracks > tracks_events = prepareTracks(
          host_tracks_pinned,
      	  host_accumulated_tracks,
      	  host_number_of_tracks_pinned,
      	  number_of_events);
      
        const bool fromNtuple = true;
        const std::string trackType = "Velo";
        callPrChecker(
	  tracks_events,
      	  folder_name_MC,
	  fromNtuple,
	  trackType);
      }
    }

    /* Plugin VeloUT CPU code here 
       Adjust input types to match PrVeloUT code
       ATTENTION: assumes we run with 1 stream only
     */

    PrVeloUT velout;
    if ( velout.initialize() ) {
      for ( int i_event = 0; i_event < number_of_events; ++i_event ) {
	// Prepare hits
	std::array<std::vector<VeloUTTracking::Hit>,VeloUTTracking::n_layers> inputHits;
	for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
	  for ( int i_hit = 0; i_hit < n_hits_layers_events[i_event][i_layer]; ++i_hit ) {
	    VeloUTTracking::Hit hit;
	    hit.m_cos = hits_layers_events[i_event].cos[i_hit];
	    hit.m_dxDy = hits_layers_events[i_event].dxDy[i_hit];
	    hit.m_weight = hits_layers_events[i_event].weight[i_hit];
	    hit.m_xAtYEq0 = hits_layers_events[i_event].xAtYEq0[i_hit];
	    hit.m_yBegin = hits_layers_events[i_event].yBegin[i_hit];
	    hit.m_yEnd = hits_layers_events[i_event].yEnd[i_hit];
	    hit.m_zAtYEq0 = hits_layers_events[i_event].zAtYEq0[i_hit];
	    
	    inputHits[i_layer].push_back( hit );
	  }
	}
	
	// Prepare Velo states
	VeloState* velo_states_event = host_velo_states + host_accumulated_tracks[i_event];
	std::vector<VeloUTTracking::TrackVelo> tracks;
	for ( uint i_track = 0; i_track < host_number_of_tracks_pinned[i_event]; i_track++ ) {
	  VeloUTTracking::TrackVelo states;
	  states.push_back( velo_states_event[i_track] );
	  tracks.push_back( states );
	}
	debug_cout << "at event " << i_event << ", pass " << tracks.size() << " velo states and " << inputHits[0].size() << " hits in layer 0 to velout" << std::endl;
	
	velout(tracks, inputHits); 
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
