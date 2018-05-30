#include "../include/Stream.cuh"

cudaError_t Stream::operator()(
  const char* host_events_pinned,
  const uint* host_event_offsets_pinned,
  size_t host_events_pinned_size,
  size_t host_event_offsets_pinned_size,
  uint start_event,
  uint number_of_events,
  uint number_of_repetitions
) {
  for (uint repetitions=0; repetitions<number_of_repetitions; ++repetitions) {
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
    // cudaCheck(cudaMemcpyAsync(&number_of_hits, dev_estimated_input_size + number_of_events * VeloTracking::n_modules, sizeof(uint), cudaMemcpyDeviceToHost, stream));

    // if (number_of_hits * 6 * sizeof(uint32_t) > velo_cluster_container_size) {
    //   warning_cout << "Number of hits: " << number_of_hits << std::endl
    //     << "Size of velo cluster container is larger than previously accomodated." << std::endl
    //     << "Resizing from " << velo_cluster_container_size << " to " << number_of_hits * 6 * sizeof(uint) << " B" << std::endl;

    //   cudaCheck(cudaFree(dev_velo_cluster_container));
    //   velo_cluster_container_size = number_of_hits * 6 * sizeof(uint32_t);
    //   cudaCheck(cudaMalloc((void**)&dev_velo_cluster_container, velo_cluster_container_size));
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


    ////////////////////////////////////////
    // Optional: Simplified Kalman filter //
    // DvB: should not be optional, at least not
    // the state at the last measurement  //
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
      
      cudaCheck(cudaMemcpyAsync(host_tracks_pinned, consolidateTracks.dev_output_tracks, number_of_events * max_tracks_in_event * sizeof(Track<do_mc_check>), cudaMemcpyDeviceToHost, stream));
      
      cudaEventRecord(cuda_generic_event, stream);
      cudaEventSynchronize(cuda_generic_event);

      // std::cout << "Number of tracks found per event:" << std::endl;
      // for (int i=0; i<number_of_events; ++i) {
      //   std::cout << i << ": " << host_number_of_tracks_pinned[i] << std::endl;
      // }
      
      // Calculating the sum on CPU (the code below) slows down the CUDA stream
      // If we do this on CPU, it should happen concurrently to some CUDA stream
      // if (do_consolidate) {
      //   // Calculate number of tracks (prefix sum) and fetch consolidated tracks
      //   int total_number_of_tracks = 0;
      //   for (int i=0; i<number_of_events; ++i) {
      //     total_number_of_tracks += host_number_of_tracks_pinned[i];
      //   }
      //   cudaCheck(cudaMemcpyAsync(host_tracks_pinned, searchByTriplet.dev_tracklets, total_number_of_tracks * sizeof(TrackHits), cudaMemcpyDeviceToHost, stream));
      // }
    }

    if (print_individual_rates) {
      t_total.stop();
      times.emplace_back("total", t_total.get());
      print_timing(number_of_events, times);
    }

    /* MC check */
#ifdef MC_CHECK
    if ( repetitions == 0 ) { // only print out tracks once
      /* Write tracks to file: track #, length, x, y, z, LHCb ID of all hits */
      std::ofstream out_file;
      out_file.open("tracks_out.txt");
      for ( uint i_event = 0; i_event < number_of_events; i_event++ ) {
      	Track<do_mc_check>* tracks_event;
      	tracks_event = host_tracks_pinned + i_event * max_tracks_in_event;
      	for ( uint i_track = 0; i_track < host_number_of_tracks_pinned[i_event]; i_track++ ) {
      	  //printTrack<do_mc_check>( tracks_event, i_track, out_file );
      	}
      }
      out_file.close();
     
      /* Tracks to be checked, save in format for checker */
      std::vector< trackChecker::Tracks > all_tracks; // all tracks from all events
      std::ofstream out_file_ids;
      out_file_ids.open("first_event_reco_tracks.txt");
      for ( uint i_event = 0; i_event < number_of_events; i_event++ ) {
      	Track<do_mc_check>* tracks_event = host_tracks_pinned + i_event * max_tracks_in_event;
	trackChecker::Tracks tracks; // all tracks within one event
	//printf("Found %u tracks in event %u \n", host_number_of_tracks_pinned[i_event], i_event);
      	for ( uint i_track = 0; i_track < host_number_of_tracks_pinned[i_event]; i_track++ ) {
	  trackChecker::Track t;
	  const Track <do_mc_check> track = tracks_event[i_track];
	  //printf("Track %u has %u hits \n", i_track, track.hitsNum);
	  for ( int i_hit = 0; i_hit < track.hitsNum; ++i_hit ) {
	    Hit <true> hit = track.hits[ i_hit ];
	    //printf("id = %u \n", (uint32_t)(hit.LHCbID) );
	    if ( i_event == 0 )
	      out_file_ids << std::hex << hit.LHCbID << std::endl;
	    //uint32_t masked_id = hit.LHCbID & 0x0FFFFFFF;
	    LHCbID lhcb_id( hit.LHCbID );
	    t.addId( lhcb_id );
	  } // hits
	  tracks.push_back( t );
	} // tracks
	
	all_tracks.push_back( tracks );
      } // events
      out_file_ids.close();

      checkTracks( all_tracks, folder_name_MC );
    }
#endif    
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
