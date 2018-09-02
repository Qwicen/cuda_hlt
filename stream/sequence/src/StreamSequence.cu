#include "Stream.cuh"

#include <iostream>
#include <fstream>

#include <random>

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

    //Catboost
    //Features generation
    std::vector<std::vector<float>> features;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < number_of_events; ++i)
    {
        std::vector<float> row;
        for (size_t j = 0; j < model_float_feature_num; ++j) {
            row.push_back(dis(mt));
        }
        features.push_back(row);
    }

    size_t binFeatureIndex = 0;
    for (size_t i = 0; i < number_of_events; i++) {
      for (const auto& ff : *ObliviousTrees->FloatFeatures()) {
        const auto floatVal = features[i][ff->Index()];
        for (const auto border : *ff->Borders()) {
          host_bin_features[binFeatureIndex] = (unsigned char)(floatVal > border);
          ++binFeatureIndex;
        }
      } 
    }

    // Set arguments size
    argument_sizes[arg::dev_tree_splits] = argen.size<arg::dev_tree_splits>(tree_num);
    argument_sizes[arg::dev_leaf_values] = argen.size<arg::dev_leaf_values>(tree_num);
    argument_sizes[arg::dev_tree_sizes] = argen.size<arg::dev_tree_sizes>(tree_num);
    argument_sizes[arg::dev_catboost_output] = argen.size<arg::dev_catboost_output>(number_of_events);
    argument_sizes[arg::dev_bin_features] = argen.size<arg::dev_bin_features>(model_bin_feature_num * number_of_events);

    // Reserve required arguments for this algorithm in the sequence
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);

    // Copy memory from host to device
    for (size_t i = 0; i < tree_num; i++) {
      int depth = host_tree_sizes[i];
      
      cudaMemcpy(host_leaf_values[i], leafValuesPtr_flat, (1 << depth)*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(host_tree_splits[i], treeSplitsPtr_flat, depth*sizeof(int), cudaMemcpyHostToDevice);
      
      leafValuesPtr_flat += (1 << depth);
      treeSplitsPtr_flat += depth;
    }
    cudaCheck(cudaMemcpyAsync(
      argen.generate<arg::dev_tree_splits>(argument_offsets),
      host_tree_splits,
      tree_num * sizeof(int*),
      cudaMemcpyHostToDevice,
      stream
    ));
    cudaCheck(cudaMemcpyAsync(
      argen.generate<arg::dev_leaf_values>(argument_offsets),
      host_leaf_values,
      tree_num * sizeof(double*),
      cudaMemcpyHostToDevice,
      stream
    ));
    cudaCheck(cudaMemcpyAsync(
      argen.generate<arg::dev_tree_sizes>(argument_offsets),
      host_tree_sizes,
      tree_num * sizeof(int),
      cudaMemcpyHostToDevice,
      stream
    ));
    cudaCheck(cudaMemcpyAsync(
      argen.generate<arg::dev_catboost_output>(argument_offsets),
      host_catboost_output,
      number_of_events * sizeof(float),
      cudaMemcpyHostToDevice,
      stream
    ));
    cudaCheck(cudaMemcpyAsync(
      argen.generate<arg::dev_bin_features>(argument_offsets),
      host_bin_features,
      binFeatureIndex * sizeof(char),
      cudaMemcpyHostToDevice,
      stream
    ));

    // Setup opts for kernel call
    sequence.item<seq::catboost_evaluator>().set_opts(dim3(number_of_events), dim3(32), stream, 32*sizeof(float));

    // Setup arguments for kernel call
    sequence.item<seq::catboost_evaluator>().set_arguments(
      argen.generate<arg::dev_tree_splits>(argument_offsets),
      argen.generate<arg::dev_leaf_values>(argument_offsets),
      argen.generate<arg::dev_tree_sizes>(argument_offsets),
      argen.generate<arg::dev_catboost_output>(argument_offsets),
      argen.generate<arg::dev_bin_features>(argument_offsets),
      tree_num,
      number_of_events,
      model_bin_feature_num
    );

    // Kernel call
    sequence.item<seq::catboost_evaluator>().invoke();

    // Retrieve result
    cudaCheck(cudaMemcpyAsync(host_catboost_output,
      argen.generate<arg::dev_catboost_output>(argument_offsets),
      argen.size<arg::dev_catboost_output>(number_of_events),
      cudaMemcpyDeviceToHost,
      stream
    ));

    // Wait to receive the result
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    // Check the output
    for (int i = 0; i < number_of_events; i++) {
      info_cout << "CATBOOST KERNEL OUTPUT: " << host_catboost_output[i] << std::endl;
    }
    info_cout << std::endl << std::endl;

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
    
  } // repititions
  return cudaSuccess;
}
