#include "Stream.cuh"

#include <iostream>
#include <fstream>

cudaError_t Stream::run_sequence(
  const uint i_stream,
  const char* host_velopix_events,
  const uint* host_velopix_event_offsets,
  const size_t host_velopix_events_size,
  const size_t host_velopix_event_offsets_size,
  VeloUTTracking::HitsSoA *host_ut_hits_events,
  const PrUTMagnetTool* host_ut_magnet_tool,
  const uint number_of_events,
  const uint number_of_repetitions
) {
  // Generate object for populating arguments
  DynamicArgumentGenerator<argument_tuple_t> argen {arguments, dev_base_pointer};

  // Sizes and offsets of arguments
  std::array<size_t, std::tuple_size<argument_tuple_t>::value> argument_sizes;
  std::array<uint, std::tuple_size<argument_tuple_t>::value> argument_offsets;

  for (uint repetition=0; repetition<number_of_repetitions; ++repetition) {
    uint sequence_step = 0;

    // Reset scheduler
    scheduler.reset();

    // Estimate input size
    // Set arguments and reserve memory
    argument_sizes[arg::dev_raw_input] = argen.size<arg::dev_raw_input>(host_velopix_events_size);
    argument_sizes[arg::dev_raw_input_offsets] = argen.size<arg::dev_raw_input_offsets>(host_velopix_event_offsets_size);
    argument_sizes[arg::dev_estimated_input_size] = argen.size<arg::dev_estimated_input_size>(number_of_events * VeloTracking::n_modules + 1);
    argument_sizes[arg::dev_module_cluster_num] = argen.size<arg::dev_module_cluster_num>(number_of_events * VeloTracking::n_modules);
    argument_sizes[arg::dev_module_candidate_num] = argen.size<arg::dev_raw_input_offsets>(number_of_events);
    argument_sizes[arg::dev_cluster_candidates] = argen.size<arg::dev_cluster_candidates>(number_of_events * VeloClustering::max_candidates_event);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    // Setup opts and arguments for kernel call
    sequence.item<seq::estimate_input_size>().set_opts(dim3(number_of_events), dim3(32, 26), stream);
    sequence.item<seq::estimate_input_size>().set_arguments(
      argen.generate<arg::dev_raw_input>(argument_offsets),
      argen.generate<arg::dev_raw_input_offsets>(argument_offsets),
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_module_cluster_num>(argument_offsets),
      argen.generate<arg::dev_module_candidate_num>(argument_offsets),
      argen.generate<arg::dev_cluster_candidates>(argument_offsets)
    );
    cudaCheck(cudaMemcpyAsync(argen.generate<arg::dev_raw_input>(argument_offsets), host_velopix_events, host_velopix_events_size, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(argen.generate<arg::dev_raw_input_offsets>(argument_offsets), host_velopix_event_offsets, host_velopix_event_offsets_size * sizeof(uint), cudaMemcpyHostToDevice, stream));
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    // Kernel call
    sequence.item<seq::estimate_input_size>().invoke();

    // Convert the estimated sizes to module hit start format (argument_offsets)
    // Set arguments and reserve memory
    argument_sizes[arg::dev_cluster_offset] = argen.size<arg::dev_cluster_offset>(number_of_events);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    // Setup sequence step
    const auto prefix_sum_blocks = (VeloTracking::n_modules * number_of_events + 511) / 512;
    sequence.item<seq::prefix_sum_reduce>().set_opts(dim3(prefix_sum_blocks), dim3(256), stream);
    sequence.item<seq::prefix_sum_reduce>().set_arguments(
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_cluster_offset>(argument_offsets),
      VeloTracking::n_modules * number_of_events
    );
    // Kernel call
    sequence.item<seq::prefix_sum_reduce>().invoke();

    // Prefix Sum Single Block
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::prefix_sum_single_block>().set_arguments(
      argen.generate<arg::dev_estimated_input_size>(argument_offsets) + VeloTracking::n_modules * number_of_events,
      argen.generate<arg::dev_cluster_offset>(argument_offsets),
      prefix_sum_blocks
    );
    sequence.item<seq::prefix_sum_single_block>().invoke();

    // Prefix sum scan
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    const auto prefix_sum_scan_blocks = prefix_sum_blocks==1 ? 1 : (prefix_sum_blocks-1);
    sequence.item<seq::prefix_sum_scan>().set_opts(dim3(prefix_sum_scan_blocks), dim3(512), stream);
    sequence.item<seq::prefix_sum_scan>().set_arguments(
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_cluster_offset>(argument_offsets),
      VeloTracking::n_modules * number_of_events
    );
    sequence.item<seq::prefix_sum_scan>().invoke();

    // Fetch the number of hits we require
    cudaCheck(cudaMemcpyAsync(host_total_number_of_velo_clusters, argen.generate<arg::dev_estimated_input_size>(argument_offsets) + number_of_events * VeloTracking::n_modules, sizeof(uint), cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    // Masked Velo clustering
    argument_sizes[arg::dev_velo_cluster_container] = argen.size<arg::dev_velo_cluster_container>(6 * host_total_number_of_velo_clusters[0]);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::masked_velo_clustering>().set_opts(dim3(number_of_events), dim3(256), stream);
    sequence.item<seq::masked_velo_clustering>().set_arguments(
      argen.generate<arg::dev_raw_input>(argument_offsets),
      argen.generate<arg::dev_raw_input_offsets>(argument_offsets),
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_module_cluster_num>(argument_offsets),
      argen.generate<arg::dev_module_candidate_num>(argument_offsets),
      argen.generate<arg::dev_cluster_candidates>(argument_offsets),
      argen.generate<arg::dev_velo_cluster_container>(argument_offsets),
      dev_velo_geometry
    );
    sequence.item<seq::masked_velo_clustering>().invoke();

    // Calculate phi and sort
    argument_sizes[arg::dev_hit_permutation] = argen.size<arg::dev_hit_permutation>(host_total_number_of_velo_clusters[0]);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::calculate_phi_and_sort>().set_opts(dim3(number_of_events), dim3(64), stream);
    sequence.item<seq::calculate_phi_and_sort>().set_arguments(
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_module_cluster_num>(argument_offsets),
      argen.generate<arg::dev_velo_cluster_container>(argument_offsets),
      argen.generate<arg::dev_hit_permutation>(argument_offsets)
    );
    sequence.item<seq::calculate_phi_and_sort>().invoke();

    // Fill candidates
    argument_sizes[arg::dev_h0_candidates] = argen.size<arg::dev_h0_candidates>(2 * host_total_number_of_velo_clusters[0]);
    argument_sizes[arg::dev_h2_candidates] = argen.size<arg::dev_h2_candidates>(2 * host_total_number_of_velo_clusters[0]);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    // Setup opts and arguments
    sequence.item<seq::fill_candidates>().set_opts(dim3(number_of_events, 48), dim3(128), stream);
    sequence.item<seq::fill_candidates>().set_arguments(
      argen.generate<arg::dev_velo_cluster_container>(argument_offsets),
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_module_cluster_num>(argument_offsets),
      argen.generate<arg::dev_h0_candidates>(argument_offsets),
      argen.generate<arg::dev_h2_candidates>(argument_offsets)
    );
    sequence.item<seq::fill_candidates>().invoke();

    // Search by triplet
    argument_sizes[arg::dev_tracks] = argen.size<arg::dev_tracks>(number_of_events * VeloTracking::max_tracks);
    argument_sizes[arg::dev_tracklets] = argen.size<arg::dev_tracklets>(number_of_events * VeloTracking::ttf_modulo);
    argument_sizes[arg::dev_tracks_to_follow] = argen.size<arg::dev_tracks_to_follow>(number_of_events * VeloTracking::ttf_modulo);
    argument_sizes[arg::dev_weak_tracks] = argen.size<arg::dev_weak_tracks>(number_of_events * VeloTracking::max_weak_tracks);
    argument_sizes[arg::dev_hit_used] = argen.size<arg::dev_hit_used>(host_total_number_of_velo_clusters[0]);
    argument_sizes[arg::dev_atomics_storage] = argen.size<arg::dev_atomics_storage>(number_of_events * VeloTracking::num_atomics);
    argument_sizes[arg::dev_rel_indices] = argen.size<arg::dev_rel_indices>(number_of_events * 2 * VeloTracking::max_numhits_in_module);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    // Setup opts and arguments
    sequence.item<seq::search_by_triplet>().set_opts(dim3(number_of_events), dim3(32), stream, 32 * sizeof(float));
    sequence.item<seq::search_by_triplet>().set_arguments(
      argen.generate<arg::dev_velo_cluster_container>(argument_offsets),
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_module_cluster_num>(argument_offsets),
      argen.generate<arg::dev_tracks>(argument_offsets),
      argen.generate<arg::dev_tracklets>(argument_offsets),
      argen.generate<arg::dev_tracks_to_follow>(argument_offsets),
      argen.generate<arg::dev_weak_tracks>(argument_offsets),
      argen.generate<arg::dev_hit_used>(argument_offsets),
      argen.generate<arg::dev_atomics_storage>(argument_offsets),
      argen.generate<arg::dev_h0_candidates>(argument_offsets),
      argen.generate<arg::dev_h2_candidates>(argument_offsets),
      argen.generate<arg::dev_rel_indices>(argument_offsets)
    );
    sequence.item<seq::search_by_triplet>().invoke();

    // Weak tracks adder
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    // Setup opts and arguments
    sequence.item<seq::weak_tracks_adder>().set_opts(dim3(number_of_events), dim3(32), stream);
    sequence.item<seq::weak_tracks_adder>().set_arguments(
      argen.generate<arg::dev_velo_cluster_container>(argument_offsets),
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_tracks>(argument_offsets),
      argen.generate<arg::dev_weak_tracks>(argument_offsets),
      argen.generate<arg::dev_hit_used>(argument_offsets),
      argen.generate<arg::dev_atomics_storage>(argument_offsets)
    );
    sequence.item<seq::weak_tracks_adder>().invoke();
    
    // Calculate prefix sum of found tracks
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::copy_and_prefix_sum_single_block>().set_arguments(
      (uint*) argen.generate<arg::dev_atomics_storage>(argument_offsets) + number_of_events*2,
      (uint*) argen.generate<arg::dev_atomics_storage>(argument_offsets),
      (uint*) argen.generate<arg::dev_atomics_storage>(argument_offsets) + number_of_events,
      number_of_events
    );
    sequence.item<seq::copy_and_prefix_sum_single_block>().invoke();

    // Fetch number of reconstructed tracks
    cudaCheck(cudaMemcpyAsync(host_number_of_reconstructed_velo_tracks, argen.generate<arg::dev_atomics_storage>(argument_offsets) + number_of_events * 2, sizeof(uint), cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);
    size_t velo_track_hit_number_size = host_number_of_reconstructed_velo_tracks[0] + 1;

    // Prefix sum of tracks hits
    // 1. Copy velo track hit number to a consecutive container
    // 2. Reduce
    // 3. Single block
    // 4. Scan

    // Copy Velo track hit number
    argument_sizes[arg::dev_velo_track_hit_number] = argen.size<arg::dev_velo_track_hit_number>(velo_track_hit_number_size);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::copy_velo_track_hit_number>().set_opts(dim3(number_of_events), dim3(512), stream);
    sequence.item<seq::copy_velo_track_hit_number>().set_arguments(
      argen.generate<arg::dev_tracks>(argument_offsets),
      argen.generate<arg::dev_atomics_storage>(argument_offsets),
      argen.generate<arg::dev_velo_track_hit_number>(argument_offsets)
    );
    sequence.item<seq::copy_velo_track_hit_number>().invoke();

    // Prefix sum: Reduce
    const size_t prefix_sum_auxiliary_array_2_size = (host_number_of_reconstructed_velo_tracks[0] + 511) / 512;
    argument_sizes[arg::dev_prefix_sum_auxiliary_array_2] = argen.size<arg::dev_prefix_sum_auxiliary_array_2>(prefix_sum_auxiliary_array_2_size);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::prefix_sum_reduce_velo_track_hit_number>().set_opts(dim3(prefix_sum_auxiliary_array_2_size), dim3(256), stream);
    sequence.item<seq::prefix_sum_reduce_velo_track_hit_number>().set_arguments(
      argen.generate<arg::dev_velo_track_hit_number>(argument_offsets),
      argen.generate<arg::dev_prefix_sum_auxiliary_array_2>(argument_offsets),
      host_number_of_reconstructed_velo_tracks[0]
    );
    sequence.item<seq::prefix_sum_reduce_velo_track_hit_number>().invoke();

    // Prefix sum: Single block
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::prefix_sum_single_block_velo_track_hit_number>().set_arguments(
      argen.generate<arg::dev_velo_track_hit_number>(argument_offsets) + host_number_of_reconstructed_velo_tracks[0],
      argen.generate<arg::dev_prefix_sum_auxiliary_array_2>(argument_offsets),
      prefix_sum_auxiliary_array_2_size
    );
    sequence.item<seq::prefix_sum_single_block_velo_track_hit_number>().invoke();

    // Prefix sum: Scan
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    const uint pss_velo_track_hit_number_opts =
      prefix_sum_auxiliary_array_2_size==1 ? 1 : (prefix_sum_auxiliary_array_2_size-1);
    sequence.item<seq::prefix_sum_scan_velo_track_hit_number>().set_opts(dim3(pss_velo_track_hit_number_opts), dim3(512), stream);
    sequence.item<seq::prefix_sum_scan_velo_track_hit_number>().set_arguments(
      argen.generate<arg::dev_velo_track_hit_number>(argument_offsets),
      argen.generate<arg::dev_prefix_sum_auxiliary_array_2>(argument_offsets),
      host_number_of_reconstructed_velo_tracks[0]
    );
    sequence.item<seq::prefix_sum_scan_velo_track_hit_number>().invoke();

    // Fetch total number of hits accumulated with all tracks
    cudaCheck(cudaMemcpyAsync(host_accumulated_number_of_hits_in_velo_tracks,
      argen.generate<arg::dev_velo_track_hit_number>(argument_offsets) + host_number_of_reconstructed_velo_tracks[0],
      sizeof(uint), cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    // Consolidate tracks
    argument_sizes[arg::dev_velo_track_hits] = argen.size<arg::dev_velo_track_hits>(host_accumulated_number_of_hits_in_velo_tracks[0]);
    argument_sizes[arg::dev_velo_states] = argen.size<arg::dev_velo_states>(host_number_of_reconstructed_velo_tracks[0]);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::consolidate_tracks>().set_opts(dim3(number_of_events), dim3(32), stream);
    sequence.item<seq::consolidate_tracks>().set_arguments(
      argen.generate<arg::dev_atomics_storage>(argument_offsets),
      argen.generate<arg::dev_tracks>(argument_offsets),
      argen.generate<arg::dev_velo_track_hit_number>(argument_offsets),
      argen.generate<arg::dev_velo_cluster_container>(argument_offsets),
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_module_cluster_num>(argument_offsets),
      argen.generate<arg::dev_velo_track_hits>(argument_offsets),
      argen.generate<arg::dev_velo_states>(argument_offsets)
    );
    sequence.item<seq::consolidate_tracks>().invoke();


    for (int i = 0; i < host_ut_number_of_raw_banks; ++i) {
      host_ut_raw_banks[i] = i;
    }
    
   

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
      cudaCheck(cudaMemcpyAsync(host_number_of_tracks, argen.generate<arg::dev_atomics_storage>(argument_offsets), number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_accumulated_tracks, argen.generate<arg::dev_atomics_storage>(argument_offsets) + number_of_events, number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_velo_track_hit_number, argen.generate<arg::dev_velo_track_hit_number>(argument_offsets), argen.size<arg::dev_velo_track_hit_number>(velo_track_hit_number_size), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_velo_track_hits, argen.generate<arg::dev_velo_track_hits>(argument_offsets), argen.size<arg::dev_velo_track_hits>(host_accumulated_number_of_hits_in_velo_tracks[0]), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_velo_states, argen.generate<arg::dev_velo_states>(argument_offsets), argen.size<arg::dev_velo_states>(host_number_of_reconstructed_velo_tracks[0]), cudaMemcpyDeviceToHost, stream)); 
    }

    // VeloUT tracking
    argument_sizes[arg::dev_ut_hits] = argen.size<arg::dev_ut_hits>(number_of_events);
    argument_sizes[arg::dev_veloUT_tracks] = argen.size<arg::dev_veloUT_tracks>(number_of_events*VeloUTTracking::max_num_tracks);
    argument_sizes[arg::dev_atomics_veloUT] = argen.size<arg::dev_atomics_veloUT>(VeloUTTracking::num_atomics*number_of_events);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    cudaCheck(cudaMemcpyAsync(argen.generate<arg::dev_ut_hits>(argument_offsets), host_ut_hits_events, number_of_events * sizeof(VeloUTTracking::HitsSoA), cudaMemcpyHostToDevice, stream ));
    sequence.item<seq::veloUT>().set_opts(dim3(number_of_events), dim3(32), stream);
    sequence.item<seq::veloUT>().set_arguments(
      argen.generate<arg::dev_ut_hits>(argument_offsets),
      argen.generate<arg::dev_atomics_storage>(argument_offsets),
      argen.generate<arg::dev_velo_track_hit_number>(argument_offsets),
      argen.generate<arg::dev_velo_track_hits>(argument_offsets),
      argen.generate<arg::dev_velo_states>(argument_offsets),
      argen.generate<arg::dev_veloUT_tracks>(argument_offsets),
      argen.generate<arg::dev_atomics_veloUT>(argument_offsets),
      dev_ut_magnet_tool );
    sequence.item<seq::veloUT>().invoke();

    // Transmission device to host
    if ( transmit_device_to_host) {
      cudaCheck(cudaMemcpyAsync(host_atomics_veloUT, argen.generate<arg::dev_atomics_veloUT>(argument_offsets), argen.size<arg::dev_atomics_veloUT>(VeloUTTracking::num_atomics*number_of_events), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_veloUT_tracks, argen.generate<arg::dev_veloUT_tracks>(argument_offsets), argen.size<arg::dev_veloUT_tracks>(number_of_events*VeloUTTracking::max_num_tracks), cudaMemcpyDeviceToHost, stream));
    }

    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    ///////////////////////
    // Monte Carlo Check //
    ///////////////////////
    
    if (do_check && i_stream == 0) {
      if (repetition == 0) { // only check efficiencies once

        /* CHECKING Velo TRACKS */
        if ( !transmit_device_to_host ) { // Fetch data
          cudaCheck(cudaMemcpyAsync(host_number_of_tracks, argen.generate<arg::dev_atomics_storage>(argument_offsets), number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
          cudaCheck(cudaMemcpyAsync(host_accumulated_tracks, argen.generate<arg::dev_atomics_storage>(argument_offsets) + number_of_events, number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
          cudaCheck(cudaMemcpyAsync(host_velo_track_hit_number, argen.generate<arg::dev_velo_track_hit_number>(argument_offsets), argen.size<arg::dev_velo_track_hit_number>(velo_track_hit_number_size), cudaMemcpyDeviceToHost, stream));
          cudaCheck(cudaMemcpyAsync(host_velo_track_hits, argen.generate<arg::dev_velo_track_hits>(argument_offsets), argen.size<arg::dev_velo_track_hits>(host_accumulated_number_of_hits_in_velo_tracks[0]), cudaMemcpyDeviceToHost, stream));
          cudaCheck(cudaMemcpyAsync(host_velo_states, argen.generate<arg::dev_velo_states>(argument_offsets), argen.size<arg::dev_velo_states>(host_number_of_reconstructed_velo_tracks[0]), cudaMemcpyDeviceToHost, stream)); 
          cudaEventRecord(cuda_generic_event, stream);
          cudaEventSynchronize(cuda_generic_event);
        }

	std::cout << "CHECKING VELO TRACKS " << std::endl; 
	
        const std::vector< trackChecker::Tracks > tracks_events = prepareTracks(
          host_velo_track_hit_number,
          reinterpret_cast<VeloTracking::Hit<true>*>(host_velo_track_hits),
      	  host_accumulated_tracks,
      	  host_number_of_tracks,
      	  number_of_events);
      
        std::string trackType = "Velo";
      	call_pr_checker (
	  tracks_events,
      	  folder_name_MC,
          start_event_offset,
    	  trackType);
      
        /* CHECKING VeloUT TRACKS */
        if ( !transmit_device_to_host ) { // Fetch data
          cudaCheck(cudaMemcpyAsync(host_atomics_veloUT, argen.generate<arg::dev_atomics_veloUT>(argument_offsets), argen.size<arg::dev_atomics_veloUT>(VeloUTTracking::num_atomics*number_of_events), cudaMemcpyDeviceToHost, stream));
          cudaCheck(cudaMemcpyAsync(host_veloUT_tracks, argen.generate<arg::dev_veloUT_tracks>(argument_offsets), argen.size<arg::dev_veloUT_tracks>(number_of_events*VeloUTTracking::max_num_tracks), cudaMemcpyDeviceToHost, stream));
        }
      
        const std::vector< trackChecker::Tracks > veloUT_tracks = prepareVeloUTTracks(
          host_veloUT_tracks,
          host_atomics_veloUT,
          number_of_events
        );  
      
        std::cout << "CHECKING VeloUT TRACKS from GPU" << std::endl;
        trackType = "VeloUT";
        call_pr_checker (
          veloUT_tracks,
          folder_name_MC,
          start_event_offset,
          trackType);                                                                            
      
        /* Run VeloUT on x86 architecture */
        if ( run_on_x86 ) {
          std::vector< trackChecker::Tracks > *ut_tracks_events = new std::vector< trackChecker::Tracks >;
        
          int rv = run_veloUT_on_CPU(
                     ut_tracks_events,
                     host_ut_hits_events,
                     host_ut_magnet_tool,
                     host_velo_states,
                     host_accumulated_tracks,
                     host_velo_track_hit_number,
                     host_velo_track_hits,
                     host_number_of_tracks,
                     number_of_events );

          if ( rv != 0 )
            continue;
          
          std::cout << "CHECKING VeloUT TRACKS from x86" << std::endl;
          trackType = "VeloUT";
          call_pr_checker (
            *ut_tracks_events,
            folder_name_MC,
            start_event_offset,
            trackType); 
          
          delete ut_tracks_events;
        }
      } // only in first repitition
    } // mc_check_enabled

    /* UT DECODING */

    // START BOARDS
    // Some boards could be NOT PRESENT
    uint32_t number_of_boards = 0;
    std::ifstream inReadout("../input/geometry/ut_boards.bin", std::ios::in | std::ios::binary);
    inReadout.read((char *) &(number_of_boards), sizeof(uint32_t));
    
    for (uint32_t i = 0; i < number_of_boards; ++i) {
      
      uint32_t number_of_sectors = 0;
      uint32_t stripsPerHybrid = 0;
      uint32_t boardID = 0;
      
      inReadout.read((char *) &(boardID),           sizeof(uint32_t));
      inReadout.read((char *) &(stripsPerHybrid),   sizeof(uint32_t));
      inReadout.read((char *) &(number_of_sectors), sizeof(uint32_t));

      host_ut_stripsPerHybrid[boardID] = stripsPerHybrid;

      if (number_of_sectors != 6) {
        info_cout << "ERROR LOADING BOARDS GEOMETRY" << std::endl;
      }
      
      for (uint32_t j = 0; j < number_of_sectors; ++j) {
        
        uint32_t layer = 0;
        uint32_t station = 0;
        uint32_t detRegion = 0;
        uint32_t sector = 0;
        uint32_t chanID = 0;
        
        inReadout.read((char *) &(layer),   sizeof(uint32_t));
        inReadout.read((char *) &(station), sizeof(uint32_t));
        inReadout.read((char *) &(detRegion), sizeof(uint32_t));
        inReadout.read((char *) &(sector),  sizeof(uint32_t));
        inReadout.read((char *) &(chanID),  sizeof(uint32_t));
        
        const uint32_t boardIndex = boardID * 6 + j;
        host_ut_expanded_channels->stations   [boardIndex] = station;
        host_ut_expanded_channels->layers     [boardIndex] = layer;
        host_ut_expanded_channels->detRegions [boardIndex] = detRegion;
        host_ut_expanded_channels->sectors    [boardIndex] = sector;
        host_ut_expanded_channels->chanIDs    [boardIndex] = chanID;
      }
    }
    inReadout.close();
    // END BOARDS


    // START GEOMETRY
    uint32_t number_of_sectors = 0;
    std::ifstream inSectors("../input/geometry/ut_geometry.bin", std::ios::in | std::ios::binary);

    inSectors.read((char *) &(number_of_sectors), sizeof(uint32_t));
    
    for (uint32_t i = 0; i < number_of_sectors; ++i) {
      
      uint32_t m_id = 0; //unused but useful to read the binary file
      uint32_t m_firstStrip = 0;
      float m_pitch = 0.f;
      float m_dxdy = 0.f;
      float m_dzdy = 0.f;
      float m_dy = 0.f;
      float m_dp0diX = 0.f;
      float m_dp0diY = 0.f;
      float m_dp0diZ = 0.f;
      float m_p0X = 0.f;
      float m_p0Y = 0.f;
      float m_p0Z = 0.f;
      float m_cosAngle = 0.f;

      inSectors.read((char *) &(m_id),          sizeof(uint32_t));
      inSectors.read((char *) &(m_dp0diX),      sizeof(float));
      inSectors.read((char *) &(m_dp0diY),      sizeof(float));
      inSectors.read((char *) &(m_dp0diZ),      sizeof(float));
      inSectors.read((char *) &(m_p0X),         sizeof(float));
      inSectors.read((char *) &(m_p0Y),         sizeof(float));
      inSectors.read((char *) &(m_p0Z),         sizeof(float));
      inSectors.read((char *) &(m_firstStrip),  sizeof(float));
      inSectors.read((char *) &(m_dxdy),        sizeof(float));
      inSectors.read((char *) &(m_dzdy),        sizeof(float));
      inSectors.read((char *) &(m_dy),          sizeof(float));
      inSectors.read((char *) &(m_cosAngle),    sizeof(float));
      inSectors.read((char *) &(m_pitch),       sizeof(float));

      
      host_ut_geometry->m_firstStrip  [i] = m_firstStrip;
      host_ut_geometry->m_pitch       [i] = m_pitch;
      host_ut_geometry->m_dxdy        [i] = m_dxdy;
      host_ut_geometry->m_dzdy        [i] = m_dzdy;
      host_ut_geometry->m_dy          [i] = m_dy;
      host_ut_geometry->m_dp0diX      [i] = m_dp0diX;
      host_ut_geometry->m_dp0diY      [i] = m_dp0diY;
      host_ut_geometry->m_dp0diZ      [i] = m_dp0diZ;
      host_ut_geometry->m_p0X         [i] = m_p0X;
      host_ut_geometry->m_p0Y         [i] = m_p0Y;
      host_ut_geometry->m_p0Z         [i] = m_p0Z;
      host_ut_geometry->m_cosAngle    [i] = m_cosAngle;
    }
    
    inSectors.close();
    // END GEOMETRY


    // START RAW EVENT
    std::ifstream inRawEvent("../input/minbias/ut_raw/0.bin", std::ios::in | std::ios::binary);
    
    uint32_t number_of_raw_banks = 0;
    std::vector<uint32_t> raw_bank_offsets;
    std::vector<uint32_t> raw_bank_data;

    inRawEvent.read((char *) &(number_of_raw_banks), sizeof(uint32_t));
  
    raw_bank_offsets.push_back(0); //first item has no offset
    for (uint32_t i = 0; i < number_of_raw_banks; ++i) {
      uint32_t offset;
      inRawEvent.read((char *) &(offset), sizeof(uint32_t));
      raw_bank_offsets.push_back(offset);
    }
    
    uint32_t maxOffset = raw_bank_offsets.back();
    for (uint32_t i = 0; i < maxOffset; ++i) {
      uint32_t data;
      inRawEvent.read((char *) &(data), sizeof(uint32_t));
      raw_bank_data.push_back(data);
    }
    
    inRawEvent.close();

    for (uint32_t i = 0; i < number_of_raw_banks; ++i) {
      host_ut_raw_banks_offsets[i] = raw_bank_offsets[i];
    }

    for (uint32_t i = 0; i < raw_bank_data.size(); ++i) {
      host_ut_raw_banks[i] = raw_bank_data[i];
    }
    // END RAW EVENT

    argument_sizes[arg::dev_ut_raw_banks] = argen.size<arg::dev_ut_raw_banks>(host_ut_max_size_raw_bank * host_ut_number_of_raw_banks);
    argument_sizes[arg::dev_ut_raw_banks_offsets] = argen.size<arg::dev_ut_raw_banks>(host_ut_number_of_raw_banks);
    argument_sizes[arg::dev_ut_stripsPerHybrid] = argen.size<arg::dev_ut_stripsPerHybrid>(ut_number_of_boards);
    argument_sizes[arg::dev_ut_expanded_channels] = argen.size<arg::dev_ut_expanded_channels>(1);
    argument_sizes[arg::dev_ut_geometry] = argen.size<arg::dev_ut_geometry>(1);
    argument_sizes[arg::dev_ut_hits_decoded] = argen.size<arg::dev_ut_hits_decoded>(1);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);


    cudaCheck(cudaMemcpyAsync(
      argen.generate<arg::dev_ut_raw_banks>(argument_offsets),
      host_ut_raw_banks,
      host_ut_max_size_raw_bank * host_ut_number_of_raw_banks * sizeof(uint32_t),
      cudaMemcpyHostToDevice,
      stream
    ));

    cudaCheck(cudaMemcpyAsync(
      argen.generate<arg::dev_ut_raw_banks_offsets>(argument_offsets),
      host_ut_raw_banks_offsets,
      host_ut_number_of_raw_banks * sizeof(uint32_t),
      cudaMemcpyHostToDevice,
      stream
    ));

    cudaCheck(cudaMemcpyAsync(
      argen.generate<arg::dev_ut_stripsPerHybrid>(argument_offsets),
      host_ut_stripsPerHybrid,
      ut_number_of_boards * sizeof(uint32_t),
      cudaMemcpyHostToDevice,
      stream
    ));

    cudaCheck(cudaMemcpyAsync(
      argen.generate<arg::dev_ut_expanded_channels>(argument_offsets),
      host_ut_expanded_channels,
      sizeof(UTExpandedChannelIDs),
      cudaMemcpyHostToDevice,
      stream
    ));

    cudaCheck(cudaMemcpyAsync(
      argen.generate<arg::dev_ut_geometry>(argument_offsets),
      host_ut_geometry,
      sizeof(UTGeometry),
      cudaMemcpyHostToDevice,
      stream
    ));

    sequence.item<seq::decode_raw_banks>().set_opts(dim3(1), dim3(host_ut_number_of_raw_banks), stream);

    sequence.item<seq::decode_raw_banks>().set_arguments(
      argen.generate<arg::dev_ut_raw_banks>(argument_offsets),
      argen.generate<arg::dev_ut_raw_banks_offsets>(argument_offsets),
      argen.generate<arg::dev_ut_stripsPerHybrid>(argument_offsets),
      argen.generate<arg::dev_ut_expanded_channels>(argument_offsets),
      argen.generate<arg::dev_ut_geometry>(argument_offsets),
      argen.generate<arg::dev_ut_hits_decoded>(argument_offsets),
      number_of_raw_banks
    );

    std::cout << "BEFORE INVOKE" << std::endl;

    sequence.item<seq::decode_raw_banks>().invoke();

    cudaCheck(cudaMemcpyAsync(
      host_ut_hits_decoded,
      argen.generate<arg::dev_ut_hits_decoded>(argument_offsets),
      argen.size<arg::dev_ut_hits_decoded>(1),
      cudaMemcpyDeviceToHost,
      stream
    ));

    // Wait to receive the result
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);


    // for (uint32_t i = 0; i < number_of_raw_banks; ++i) {
    //   std::cout << "[" << i << "] = " << host_ut_number_of_hits[i] << "\tsource: " << host_ut_sourceIDs[i] << "\tchan: " << host_ut_channelIDs[i] << std::endl;
    // }

    for (uint32_t i = 0; i < 20; ++i) {
      std::cout << "\nUTHit {"
      << "\n  ut_cos\t"            << host_ut_hits_decoded->m_cos          [i] 
      << "\n  ut_yBegin:\t"        << host_ut_hits_decoded->m_yBegin       [i]
      << "\n  ut_yEnd:\t"          << host_ut_hits_decoded->m_yEnd         [i]
      << "\n  ut_zAtYEq0:\t"       << host_ut_hits_decoded->m_zAtYEq0      [i]
      << "\n  ut_xAtYEq0:\t"       << host_ut_hits_decoded->m_xAtYEq0      [i]
      << "\n  ut_weight:\t"        << host_ut_hits_decoded->m_weight       [i]
      << "\n  ut_highThreshold:\t" << host_ut_hits_decoded->m_highThreshold[i]
      << "\n  ut_LHCbID:\t"        << host_ut_hits_decoded->m_LHCbID       [i]
      << "\n  ut_planeCode:\t"     << host_ut_hits_decoded->m_planeCode    [i]
      << "\n}\n";
    }


    // Check the output
    info_cout << "decode_raw_banks finished" << std::endl << std::endl;  
    
  } // repititions
  return cudaSuccess;
}
