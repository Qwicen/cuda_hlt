#include "../include/Stream.cuh"
#include "../../../main/include/Common.h"

#include "../../../PrVeloUT/src/PrVeloUT.h"


#include "TH1D.h"
#include "TFile.h"
#include "TTree.h"

cudaError_t Stream::operator()(
  const char* host_events_pinned,
  const uint* host_event_offsets_pinned,
  size_t host_events_pinned_size,
  size_t host_event_offsets_pinned_size,
  const VeloUTTracking::HitsSoA *hits_layers_events,
  const uint32_t n_hits_layers_events[][VeloUTTracking::n_layers],
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

	info_cout << "CHECKING VELO TRACKS " << std::endl;
	
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
      PrVeloUT velout;
      std::vector< trackChecker::Tracks > *ut_tracks_events = new std::vector< trackChecker::Tracks >;
      
      // Histograms only for checking and debugging
      TFile *f = new TFile("../output/veloUT.root", "RECREATE");
      TTree *t_ut_hits = new TTree("ut_hits","ut_hits");
      TTree *t_velo_states = new TTree("velo_states", "velo_states");
      TTree *t_track_hits = new TTree("track_hits", "track_hits");
      TTree *t_veloUT_tracks = new TTree("veloUT_tracks", "veloUT_tracks");
      float cos, yBegin, yEnd, dxDy, zAtYEq0, xAtYEq0, weight;
      float x, y, tx, ty, chi2, z, drdz;
      unsigned int LHCbID;
      int highThreshold, layer;
      int backward;
      float x_hit, y_hit, z_hit;
      float first_x, first_y, first_z;
      float last_x, last_y, last_z;
      float qop;
      
      t_ut_hits->Branch("cos", &cos);
      t_ut_hits->Branch("yBegin", &yBegin);
      t_ut_hits->Branch("yEnd", &yEnd);
      t_ut_hits->Branch("dxDy", &dxDy);
      t_ut_hits->Branch("zAtYEq0", &zAtYEq0);
      t_ut_hits->Branch("xAtYEq0", &xAtYEq0);
      t_ut_hits->Branch("weight", &weight);
      t_ut_hits->Branch("LHCbID", &LHCbID);
      t_ut_hits->Branch("highThreshold", &highThreshold);
      t_ut_hits->Branch("layer", &layer);
      t_velo_states->Branch("x", &x);
      t_velo_states->Branch("y", &y);
      t_velo_states->Branch("tx", &tx);
      t_velo_states->Branch("ty", &ty);
      t_velo_states->Branch("chi2", &chi2);
      t_velo_states->Branch("z", &z);
      t_velo_states->Branch("backward", &backward);
      t_velo_states->Branch("drdz", &drdz);
      t_track_hits->Branch("x", &x_hit);
      t_track_hits->Branch("y", &y_hit);
      t_track_hits->Branch("z", &z_hit);
      t_velo_states->Branch("first_x", &first_x);
      t_velo_states->Branch("first_y", &first_y);
      t_velo_states->Branch("first_z", &first_z); 
      t_velo_states->Branch("last_x", &last_x);
      t_velo_states->Branch("last_y", &last_y);
      t_velo_states->Branch("last_z", &last_z); 
      t_veloUT_tracks->Branch("qop", &qop);
      
      if ( velout.initialize() ) {
    	for ( int i_event = 0; i_event < number_of_events; ++i_event ) {
	  // find out offsets for every layer
	  int accumulated_hits = 0;
	  int accumulated_hits_layers[4];
	  for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
	    accumulated_hits_layers[i_layer] = accumulated_hits;
	    accumulated_hits += n_hits_layers_events[i_event][i_layer];
	  }
	  // Prepare hits
	  std::array<std::vector<VeloUTTracking::Hit>,VeloUTTracking::n_layers> inputHits;
	  for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
	    inputHits[i_layer].clear();
	    int layer_offset = accumulated_hits_layers[i_layer];
    	    for ( int i_hit = 0; i_hit < n_hits_layers_events[i_event][i_layer]; ++i_hit ) {
    	      VeloUTTracking::Hit hit;
    	      hit.m_cos = hits_layers_events[i_event].cos[layer_offset + i_hit];
    	      hit.m_dxDy = hits_layers_events[i_event].dxDy[layer_offset + i_hit];
    	      hit.m_weight = hits_layers_events[i_event].weight[layer_offset + i_hit];
    	      hit.m_xAtYEq0 = hits_layers_events[i_event].xAtYEq0[layer_offset + i_hit];
    	      hit.m_yBegin = hits_layers_events[i_event].yBegin[layer_offset + i_hit];
    	      hit.m_yEnd = hits_layers_events[i_event].yEnd[layer_offset + i_hit];
    	      hit.m_zAtYEq0 = hits_layers_events[i_event].zAtYEq0[layer_offset + i_hit];
    	      hit.m_LHCbID = hits_layers_events[i_event].LHCbID[layer_offset + i_hit];
	      hit.m_planeCode = i_layer;
	      
    	      inputHits[i_layer].push_back( hit );

    	      // For tree filling
    	      cos = hit.m_cos;
    	      yBegin = hit.m_yBegin;
    	      yEnd = hit.m_yEnd;
    	      dxDy = hit.m_dxDy;
    	      zAtYEq0 = hit.m_zAtYEq0;
    	      xAtYEq0 = hit.m_xAtYEq0;
    	      weight = hit.m_weight;
    	      LHCbID = hit.m_LHCbID;
	      layer = i_layer;
	      
    	      t_ut_hits->Fill();
    	    }
    	    // sort hits according to xAtYEq0
    	    std::sort( inputHits[i_layer].begin(), inputHits[i_layer].end(), [](VeloUTTracking::Hit a, VeloUTTracking::Hit b) { return a.xAtYEq0() < b.xAtYEq0(); } );
    	  }
	  
    	  // Prepare Velo tracks
    	  VeloState* velo_states_event = host_velo_states + host_accumulated_tracks[i_event];
    	  VeloTracking::Track<true>* tracks_event = host_tracks_pinned + host_accumulated_tracks[i_event];
    	  std::vector<VeloUTTracking::TrackVelo> tracks;
	  for ( uint i_track = 0; i_track < host_number_of_tracks_pinned[i_event]; i_track++ ) {

    	    VeloUTTracking::TrackVelo track;

    	    VeloUTTracking::TrackUT ut_track;
    	    const VeloTracking::Track<true> velo_track = tracks_event[i_track];
    	    backward = (int)velo_track.backward;
    	    ut_track.hitsNum = velo_track.hitsNum;
    	    for ( int i_hit = 0; i_hit < velo_track.hitsNum; ++i_hit ) {
    	      ut_track.LHCbIDs.push_back( velo_track.hits[i_hit].LHCbID );
    	    }
    	    track.track = ut_track;
	    
    	    track.state = ( velo_states_event[i_track] );

	    //////////////////////
    	    // For tree filling
	    //////////////////////
    	    x = track.state.x;
    	    y = track.state.y;
    	    tx = track.state.tx;
    	    ty = track.state.ty;
    	    chi2 = track.state.chi2;
    	    z = track.state.z;
	    // study (sign of) (dr/dz) -> track moving away from beamline?
    	    // drop 1/sqrt(x^2+y^2) to avoid sqrt calculation, no effect on sign
    	    float dx = velo_track.hits[velo_track.hitsNum - 1].x - velo_track.hits[0].x;
    	    float dy = velo_track.hits[velo_track.hitsNum - 1].y - velo_track.hits[0].y;
    	    float dz = velo_track.hits[velo_track.hitsNum - 1].z - velo_track.hits[0].z;
    	    drdz = velo_track.hits[0].x * dx/dz + velo_track.hits[0].y * dy/dz;

	    first_x = velo_track.hits[0].x;
	    first_y = velo_track.hits[0].y;
	    first_z = velo_track.hits[0].z;
	    last_x = velo_track.hits[velo_track.hitsNum-1].x;
	    last_y = velo_track.hits[velo_track.hitsNum-1].y;
	    last_z = velo_track.hits[velo_track.hitsNum-1].z;
	    
    	    t_velo_states->Fill();

	    /* Get hits on track */
	    for ( int i_hit = 0; i_hit < velo_track.hitsNum; ++i_hit ) {
	      x_hit = velo_track.hits[i_hit].x;
	      y_hit = velo_track.hits[i_hit].y;
	      z_hit = velo_track.hits[i_hit].z;

	      t_track_hits->Fill();
	    }
	    
	    
    	    if ( velo_track.backward ) continue;
    	    tracks.push_back( track );
    	  }
    	  debug_cout << "at event " << i_event << ", pass " << tracks.size() << " tracks and " << inputHits[0].size() << " hits in layer 0, " << inputHits[1].size() << " hits in layer 1, " << inputHits[2].size() << " hits in layer 2, " << inputHits[3].size() << " in layer 3 to velout" << std::endl;
	  
    	  std::vector< VeloUTTracking::TrackUT > ut_tracks = velout(tracks, inputHits);
    	  debug_cout << "\t got " << (uint)ut_tracks.size() << " tracks from VeloUT " << std::endl;

	  // store qop in tree
	  for ( auto veloUT_track : ut_tracks ) {
	    qop = veloUT_track.qop;
	    t_veloUT_tracks->Fill();
	  }
	  
	  // save in format for track checker
	  trackChecker::Tracks checker_tracks = prepareVeloUTTracks( ut_tracks );
	  debug_cout << "Passing " << checker_tracks.size() << " tracks to PrChecker" << std::endl;
	  int i_track = 0;
	  for ( auto ch_track : checker_tracks ) {
	    debug_cout << "\t at track " << i_track << std::endl;
	    i_track++;
	    for ( auto id : ch_track.ids() ) {
	      debug_cout << "\t id = " << uint32_t(id)   << std::endl;
	    }
	  }
	  ut_tracks_events->emplace_back( checker_tracks );
	  
    	}
	
	info_cout << "CHECKING VeloUT TRACKS" << std::endl;
	const bool fromNtuple = true;
        const std::string trackType = "VeloUT";
        call_pr_checker (
    	  *ut_tracks_events,
      	  folder_name_MC,
    	  fromNtuple,
    	  trackType); 
	
    	f->Write();
    	f->Close();
	
	delete ut_tracks_events;
      }
      
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
