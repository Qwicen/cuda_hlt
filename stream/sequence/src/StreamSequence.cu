#include "Stream.cuh"

#include <iostream>
#include <fstream>
#include <iomanip>

cudaError_t Stream::run_sequence(
  const uint i_stream,
  const RuntimeOptions& runtime_options
) {
  // Sequence tuple
  const sequence_tuple_n sequence_tuple;

  for (uint repetition=0; repetition<runtime_options.number_of_repetitions; ++repetition) {
    // Generate object for populating arguments
    ArgumentManager<argument_tuple_t> arguments {dev_base_pointer};

    // Reset scheduler
    scheduler.reset();

    // For when we have C++17
    // state_n state;
    // state = std::visit(*this, state, arguments, runtime_options);
    // state = std::visit(*this, state, arguments, runtime_options);

    // Non-C++17 solution

    // Visit all algorithms in configured sequence
    run_sequence_tuple(*this, sequence_tuple, arguments, runtime_options);

    // // Convert the estimated sizes to module hit start format (argument_offsets)
    // // Set arguments and reserve memory
    // arguments.set_size<arg::dev_cluster_offset>(number_of_events);
    // scheduler.setup_next(arguments, sequence_step++);
    // // Setup sequence step
    // const auto prefix_sum_blocks = (VeloTracking::n_modules * number_of_events + 511) / 512;
    // sequence.set_opts<seq::prefix_sum_reduce>(dim3(prefix_sum_blocks), dim3(256), stream);
    // sequence.set_arguments<seq::prefix_sum_reduce>(
    //   arguments.offset<arg::dev_estimated_input_size>(),
    //   arguments.offset<arg::dev_cluster_offset>(),
    //   VeloTracking::n_modules * number_of_events
    // );
    // // Kernel call
    // sequence.invoke<seq::prefix_sum_reduce>();

    // // Prefix Sum Single Block
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_arguments<seq::prefix_sum_single_block>(
    //   arguments.offset<arg::dev_estimated_input_size>() + VeloTracking::n_modules * number_of_events,
    //   arguments.offset<arg::dev_cluster_offset>(),
    //   prefix_sum_blocks
    // );
    // sequence.invoke<seq::prefix_sum_single_block>();

    // // Prefix sum scan
    // scheduler.setup_next(arguments, sequence_step++);
    // const auto prefix_sum_scan_blocks = prefix_sum_blocks==1 ? 1 : (prefix_sum_blocks-1);
    // sequence.set_opts<seq::prefix_sum_scan>(dim3(prefix_sum_scan_blocks), dim3(512), stream);
    // sequence.set_arguments<seq::prefix_sum_scan>(
    //   arguments.offset<arg::dev_estimated_input_size>(),
    //   arguments.offset<arg::dev_cluster_offset>(),
    //   VeloTracking::n_modules * number_of_events
    // );
    // sequence.invoke<seq::prefix_sum_scan>();

    // // Fetch the number of hits we require
    // cudaCheck(cudaMemcpyAsync(host_total_number_of_velo_clusters, arguments.offset<arg::dev_estimated_input_size>() + number_of_events * VeloTracking::n_modules, sizeof(uint), cudaMemcpyDeviceToHost, stream));
    // cudaEventRecord(cuda_generic_event, stream);
    // cudaEventSynchronize(cuda_generic_event);

    // // Masked Velo clustering
    // arguments.set_size<arg::dev_velo_cluster_container>(6 * host_total_number_of_velo_clusters[0]);
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_opts<seq::masked_velo_clustering>(dim3(number_of_events), dim3(256), stream);
    // sequence.set_arguments<seq::masked_velo_clustering>(
    //   arguments.offset<arg::dev_raw_input>(),
    //   arguments.offset<arg::dev_raw_input_offsets>(),
    //   arguments.offset<arg::dev_estimated_input_size>(),
    //   arguments.offset<arg::dev_module_cluster_num>(),
    //   arguments.offset<arg::dev_module_candidate_num>(),
    //   arguments.offset<arg::dev_cluster_candidates>(),
    //   arguments.offset<arg::dev_velo_cluster_container>(),
    //   dev_velo_geometry,
    //   constants.dev_velo_sp_patterns,
    //   constants.dev_velo_sp_fx,
    //   constants.dev_velo_sp_fy
    // );
    // sequence.invoke<seq::masked_velo_clustering>();

    // // Calculate phi and sort
    // arguments.set_size<arg::dev_hit_permutation>(host_total_number_of_velo_clusters[0]);
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_opts<seq::calculate_phi_and_sort>(dim3(number_of_events), dim3(64), stream);
    // sequence.set_arguments<seq::calculate_phi_and_sort>(
    //   arguments.offset<arg::dev_estimated_input_size>(),
    //   arguments.offset<arg::dev_module_cluster_num>(),
    //   arguments.offset<arg::dev_velo_cluster_container>(),
    //   arguments.offset<arg::dev_hit_permutation>()
    // );
    // sequence.invoke<seq::calculate_phi_and_sort>();

    // // Fill candidates
    // arguments.set_size<arg::dev_h0_candidates>(2 * host_total_number_of_velo_clusters[0]);
    // arguments.set_size<arg::dev_h2_candidates>(2 * host_total_number_of_velo_clusters[0]);
    // scheduler.setup_next(arguments, sequence_step++);
    // // Setup opts and arguments
    // sequence.set_opts<seq::fill_candidates>(dim3(number_of_events, 48), dim3(128), stream);
    // sequence.set_arguments<seq::fill_candidates>(
    //   arguments.offset<arg::dev_velo_cluster_container>(),
    //   arguments.offset<arg::dev_estimated_input_size>(),
    //   arguments.offset<arg::dev_module_cluster_num>(),
    //   arguments.offset<arg::dev_h0_candidates>(),
    //   arguments.offset<arg::dev_h2_candidates>()
    // );
    // sequence.invoke<seq::fill_candidates>();

    // // Search by triplet
    // arguments.set_size<arg::dev_tracks>(number_of_events * VeloTracking::max_tracks);
    // arguments.set_size<arg::dev_tracklets>(number_of_events * VeloTracking::ttf_modulo);
    // arguments.set_size<arg::dev_tracks_to_follow>(number_of_events * VeloTracking::ttf_modulo);
    // arguments.set_size<arg::dev_weak_tracks>(number_of_events * VeloTracking::max_weak_tracks);
    // arguments.set_size<arg::dev_hit_used>(host_total_number_of_velo_clusters[0]);
    // arguments.set_size<arg::dev_atomics_storage>(number_of_events * VeloTracking::num_atomics);
    // arguments.set_size<arg::dev_rel_indices>(number_of_events * 2 * VeloTracking::max_numhits_in_module);
    // scheduler.setup_next(arguments, sequence_step++);
    // // Setup opts and arguments
    // sequence.set_opts<seq::search_by_triplet>(dim3(number_of_events), dim3(32), stream, 32 * sizeof(float));
    // sequence.set_arguments<seq::search_by_triplet>(
    //   arguments.offset<arg::dev_velo_cluster_container>(),
    //   arguments.offset<arg::dev_estimated_input_size>(),
    //   arguments.offset<arg::dev_module_cluster_num>(),
    //   arguments.offset<arg::dev_tracks>(),
    //   arguments.offset<arg::dev_tracklets>(),
    //   arguments.offset<arg::dev_tracks_to_follow>(),
    //   arguments.offset<arg::dev_weak_tracks>(),
    //   arguments.offset<arg::dev_hit_used>(),
    //   arguments.offset<arg::dev_atomics_storage>(),
    //   arguments.offset<arg::dev_h0_candidates>(),
    //   arguments.offset<arg::dev_h2_candidates>(),
    //   arguments.offset<arg::dev_rel_indices>(),
    //   constants.dev_velo_module_zs
    // );
    // sequence.invoke<seq::search_by_triplet>();

    // // Weak tracks adder
    // scheduler.setup_next(arguments, sequence_step++);
    // // Setup opts and arguments
    // sequence.set_opts<seq::weak_tracks_adder>(dim3(number_of_events), dim3(256), stream);
    // sequence.set_arguments<seq::weak_tracks_adder>(
    //   arguments.offset<arg::dev_velo_cluster_container>(),
    //   arguments.offset<arg::dev_estimated_input_size>(),
    //   arguments.offset<arg::dev_tracks>(),
    //   arguments.offset<arg::dev_weak_tracks>(),
    //   arguments.offset<arg::dev_hit_used>(),
    //   arguments.offset<arg::dev_atomics_storage>()
    // );
    // sequence.invoke<seq::weak_tracks_adder>();

    // // Calculate prefix sum of found tracks
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_arguments<seq::copy_and_prefix_sum_single_block>(
    //   (uint*) arguments.offset<arg::dev_atomics_storage>() + number_of_events*2,
    //   (uint*) arguments.offset<arg::dev_atomics_storage>(),
    //   (uint*) arguments.offset<arg::dev_atomics_storage>() + number_of_events,
    //   number_of_events
    // );
    // sequence.invoke<seq::copy_and_prefix_sum_single_block>();

    // // Fetch number of reconstructed tracks
    // cudaCheck(cudaMemcpyAsync(host_number_of_reconstructed_velo_tracks, arguments.offset<arg::dev_atomics_storage>() + number_of_events * 2, sizeof(uint), cudaMemcpyDeviceToHost, stream));
    // cudaEventRecord(cuda_generic_event, stream);
    // cudaEventSynchronize(cuda_generic_event);
    // size_t velo_track_hit_number_size = host_number_of_reconstructed_velo_tracks[0] + 1;

    // // Prefix sum of tracks hits
    // // 1. Copy velo track hit number to a consecutive container
    // // 2. Reduce
    // // 3. Single block
    // // 4. Scan

    // // Copy Velo track hit number
    // arguments.set_size<arg::dev_velo_track_hit_number>(velo_track_hit_number_size);
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_opts<seq::copy_velo_track_hit_number>(dim3(number_of_events), dim3(512), stream);
    // sequence.set_arguments<seq::copy_velo_track_hit_number>(
    //   arguments.offset<arg::dev_tracks>(),
    //   arguments.offset<arg::dev_atomics_storage>(),
    //   arguments.offset<arg::dev_velo_track_hit_number>()
    // );
    // sequence.invoke<seq::copy_velo_track_hit_number>();

    // // Prefix sum: Reduce
    // const size_t prefix_sum_auxiliary_array_2_size = (host_number_of_reconstructed_velo_tracks[0] + 511) / 512;
    // arguments.set_size<arg::dev_prefix_sum_auxiliary_array_2>(prefix_sum_auxiliary_array_2_size);
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_opts<seq::prefix_sum_reduce_velo_track_hit_number>(dim3(prefix_sum_auxiliary_array_2_size), dim3(256), stream);
    // sequence.set_arguments<seq::prefix_sum_reduce_velo_track_hit_number>(
    //   arguments.offset<arg::dev_velo_track_hit_number>(),
    //   arguments.offset<arg::dev_prefix_sum_auxiliary_array_2>(),
    //   host_number_of_reconstructed_velo_tracks[0]
    // );
    // sequence.invoke<seq::prefix_sum_reduce_velo_track_hit_number>();

    // // Prefix sum: Single block
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_arguments<seq::prefix_sum_single_block_velo_track_hit_number>(
    //   arguments.offset<arg::dev_velo_track_hit_number>() + host_number_of_reconstructed_velo_tracks[0],
    //   arguments.offset<arg::dev_prefix_sum_auxiliary_array_2>(),
    //   prefix_sum_auxiliary_array_2_size
    // );
    // sequence.invoke<seq::prefix_sum_single_block_velo_track_hit_number>();

    // // Prefix sum: Scan
    // scheduler.setup_next(arguments, sequence_step++);
    // const uint pss_velo_track_hit_number_opts =
    //   prefix_sum_auxiliary_array_2_size==1 ? 1 : (prefix_sum_auxiliary_array_2_size-1);
    // sequence.set_opts<seq::prefix_sum_scan_velo_track_hit_number>(dim3(pss_velo_track_hit_number_opts), dim3(512), stream);
    // sequence.set_arguments<seq::prefix_sum_scan_velo_track_hit_number>(
    //   arguments.offset<arg::dev_velo_track_hit_number>(),
    //   arguments.offset<arg::dev_prefix_sum_auxiliary_array_2>(),
    //   host_number_of_reconstructed_velo_tracks[0]
    // );
    // sequence.invoke<seq::prefix_sum_scan_velo_track_hit_number>();

    // // Fetch total number of hits accumulated with all tracks
    // cudaCheck(cudaMemcpyAsync(host_accumulated_number_of_hits_in_velo_tracks,
    //   arguments.offset<arg::dev_velo_track_hit_number>() + host_number_of_reconstructed_velo_tracks[0],
    //   sizeof(uint), cudaMemcpyDeviceToHost, stream));
    // cudaEventRecord(cuda_generic_event, stream);
    // cudaEventSynchronize(cuda_generic_event);

    // // Consolidate tracks
    // // TODO: The size specified (sizeof(Hits) / sizeof(uint)) is due to the
    // //       lgenfe error from the nvcc compiler, present in Cuda 9.2. Once it
    // //       is gone, we can switch all pointers to char*.
    // arguments.set_size<arg::dev_velo_track_hits>(host_accumulated_number_of_hits_in_velo_tracks[0] * sizeof(Velo::Hit) / sizeof(uint));
    // arguments.set_size<arg::dev_velo_states>(host_number_of_reconstructed_velo_tracks[0] * sizeof(Velo::State) / sizeof(uint));
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_opts<seq::consolidate_tracks>(dim3(number_of_events), dim3(256), stream);
    // sequence.set_arguments<seq::consolidate_tracks>(
    //   arguments.offset<arg::dev_atomics_storage>(),
    //   arguments.offset<arg::dev_tracks>(),
    //   arguments.offset<arg::dev_velo_track_hit_number>(),
    //   arguments.offset<arg::dev_velo_cluster_container>(),
    //   arguments.offset<arg::dev_estimated_input_size>(),
    //   arguments.offset<arg::dev_module_cluster_num>(),
    //   arguments.offset<arg::dev_velo_track_hits>(),
    //   arguments.offset<arg::dev_velo_states>()
    // );
    // sequence.invoke<seq::consolidate_tracks>();

    // // Calculate number of UT hits
    // // Set arguments and reserve memory
    // arguments.set_size<arg::dev_ut_raw_input>(host_ut_events_size);
    // arguments.set_size<arg::dev_ut_raw_input_offsets>(host_ut_event_offsets_size);
    // arguments.set_size<arg::dev_ut_hit_offsets>(number_of_events * constants.host_unique_x_sector_layer_offsets[4] + 1);
    // scheduler.setup_next(arguments, sequence_step++);
    // // Setup opts and arguments for kernel call
    // cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_ut_raw_input>(), host_ut_events, host_ut_events_size, cudaMemcpyHostToDevice, stream));
    // cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_ut_raw_input_offsets>(), host_ut_event_offsets, host_ut_event_offsets_size * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    // cudaEventRecord(cuda_generic_event, stream);
    // cudaEventSynchronize(cuda_generic_event);
    // sequence.set_opts<seq::ut_calculate_number_of_hits>(dim3(number_of_events), dim3(192, 2), stream);
    // sequence.set_arguments<seq::ut_calculate_number_of_hits>(
    //   arguments.offset<arg::dev_ut_raw_input>(),
    //   arguments.offset<arg::dev_ut_raw_input_offsets>(),
    //   dev_ut_boards,
    //   constants.dev_ut_region_offsets,
    //   constants.dev_unique_x_sector_layer_offsets,
    //   constants.dev_unique_x_sector_offsets,
    //   arguments.offset<arg::dev_ut_hit_offsets>()
    // );
    // // Invoke kernel
    // sequence.invoke<seq::ut_calculate_number_of_hits>();

    // // // Print UT hit count per event per layer
    // // std::vector<uint> host_ut_hit_count (number_of_events * constants.host_unique_x_sector_layer_offsets[4] + 1);
    // // cudaCheck(cudaMemcpyAsync(host_ut_hit_count.data(), arguments.offset<arg::dev_ut_hit_offsets>(), argen.size<arg::dev_ut_hit_offsets>(number_of_events * constants.host_unique_x_sector_layer_offsets[4] + 1), cudaMemcpyDeviceToHost, stream));
    // // cudaEventRecord(cuda_generic_event, stream);
    // // cudaEventSynchronize(cuda_generic_event);
    // // for (int e=0; e<number_of_events; ++e) {
    // //   info_cout << "Event " << e << ", #hits per layer: ";
    // //   uint32_t* count = host_ut_hit_count.data() + e * constants.host_unique_x_sector_layer_offsets[4];
    // //   for (uint32_t i = 0; i < constants.host_unique_x_sector_layer_offsets[4]; ++i) {
    // //     info_cout << count[i] << ", ";
    // //   }
    // //   info_cout << std::endl;
    // // }

    // // Prefix sum of hit count (becomes hit offset)
    // // 1. Reduce
    // // 2. Single block
    // // 3. Scan

    // // Prefix sum: Reduce
    // const uint total_number_of_sectors = number_of_events * constants.host_unique_x_sector_layer_offsets[4];
    // const size_t prefix_sum_auxiliary_array_3_size = (total_number_of_sectors + 511) / 512;
    // arguments.set_size<arg::dev_prefix_sum_auxiliary_array_3>(prefix_sum_auxiliary_array_3_size);
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_opts<seq::prefix_sum_reduce_ut_hits>(dim3(prefix_sum_auxiliary_array_3_size), dim3(256), stream);
    // sequence.set_arguments<seq::prefix_sum_reduce_ut_hits>(
    //   arguments.offset<arg::dev_ut_hit_offsets>(),
    //   arguments.offset<arg::dev_prefix_sum_auxiliary_array_3>(),
    //   total_number_of_sectors
    // );
    // sequence.invoke<seq::prefix_sum_reduce_ut_hits>();

    // // Prefix sum: Single block
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_arguments<seq::prefix_sum_single_block_ut_hits>(
    //   arguments.offset<arg::dev_ut_hit_offsets>() + total_number_of_sectors,
    //   arguments.offset<arg::dev_prefix_sum_auxiliary_array_3>(),
    //   prefix_sum_auxiliary_array_3_size
    // );
    // sequence.invoke<seq::prefix_sum_single_block_ut_hits>();

    // // Prefix sum: Scan
    // scheduler.setup_next(arguments, sequence_step++);
    // const uint pss_ut_hits_blocks = prefix_sum_auxiliary_array_3_size==1 ? 1 : (prefix_sum_auxiliary_array_3_size-1);
    // sequence.set_opts<seq::prefix_sum_scan_ut_hits>(dim3(pss_ut_hits_blocks), dim3(512), stream);
    // sequence.set_arguments<seq::prefix_sum_scan_ut_hits>(
    //   arguments.offset<arg::dev_ut_hit_offsets>(),
    //   arguments.offset<arg::dev_prefix_sum_auxiliary_array_3>(),
    //   total_number_of_sectors
    // );
    // sequence.invoke<seq::prefix_sum_scan_ut_hits>();

    // // Fetch total number of hits accumulated with all tracks
    // cudaCheck(cudaMemcpyAsync(host_accumulated_number_of_ut_hits,
    //   arguments.offset<arg::dev_ut_hit_offsets>() + total_number_of_sectors,
    //   sizeof(uint), cudaMemcpyDeviceToHost, stream));
    // cudaEventRecord(cuda_generic_event, stream);
    // cudaEventSynchronize(cuda_generic_event);

    // // // Now, we should have the offset instead, and the sum of all in host_accumulated_number_of_ut_hits
    // // // Check that
    // // cudaCheck(cudaMemcpyAsync(host_ut_hit_count.data(), arguments.offset<arg::dev_ut_hit_offsets>(), arguments.size<arg::dev_ut_hit_offsets>(), cudaMemcpyDeviceToHost, stream));
    // // cudaEventRecord(cuda_generic_event, stream);
    // // cudaEventSynchronize(cuda_generic_event);
    // // for (int e=0; e<number_of_events; ++e) {
    // //   info_cout << "Event " << e << ", offset per sector group: ";
    // //   uint32_t* offset = host_ut_hit_count.data() + e * constants.host_unique_x_sector_layer_offsets[4];
    // //   for (uint32_t i = 0; i < constants.host_unique_x_sector_layer_offsets[4]; ++i) {
    // //     info_cout << offset[i] << ", ";
    // //   }
    // //   info_cout << std::endl;
    // // }
    // // info_cout << "Total number of UT hits: " << *host_accumulated_number_of_ut_hits << std::endl;

    // // Decode UT raw banks
    // arguments.set_size<arg::dev_ut_hits>(UTHits::number_of_arrays * host_accumulated_number_of_ut_hits[0]);
    // arguments.set_size<arg::dev_ut_hit_count>(number_of_events * constants.host_unique_x_sector_layer_offsets[4]);
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_opts<seq::decode_raw_banks>(dim3(number_of_events), dim3(64, 4), stream);
    // sequence.set_arguments<seq::decode_raw_banks>(
    //   arguments.offset<arg::dev_ut_raw_input>(),
    //   arguments.offset<arg::dev_ut_raw_input_offsets>(),
    //   dev_ut_boards,
    //   dev_ut_geometry,
    //   constants.dev_ut_region_offsets,
    //   constants.dev_unique_x_sector_layer_offsets,
    //   constants.dev_unique_x_sector_offsets,
    //   arguments.offset<arg::dev_ut_hit_offsets>(),
    //   arguments.offset<arg::dev_ut_hits>(),
    //   arguments.offset<arg::dev_ut_hit_count>()
    // );
    // sequence.invoke<seq::decode_raw_banks>();
    
    // // UT hit sorting by y
    // arguments.set_size<arg::dev_ut_hit_permutations>(host_accumulated_number_of_ut_hits[0]);
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_opts<seq::sort_by_y>(dim3(number_of_events), dim3(256), stream);
    // sequence.set_arguments<seq::sort_by_y>(
    //   arguments.offset<arg::dev_ut_hits>(),
    //   arguments.offset<arg::dev_ut_hit_offsets>(),
    //   arguments.offset<arg::dev_ut_hit_permutations>(),
    //   constants.dev_unique_x_sector_layer_offsets,
    //   constants.dev_unique_x_sector_offsets,
    //   constants.dev_unique_sector_xs
    // );
    // sequence.invoke<seq::sort_by_y>();
    
    // // VeloUT tracking
    // arguments.set_size<arg::dev_veloUT_tracks>(number_of_events * VeloUTTracking::max_num_tracks);
    // arguments.set_size<arg::dev_atomics_veloUT>(VeloUTTracking::num_atomics * number_of_events);
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_opts<seq::veloUT>(dim3(number_of_events), dim3(32), stream);
    // sequence.set_arguments<seq::veloUT>(
    //   arguments.offset<arg::dev_ut_hits>(),
    //   arguments.offset<arg::dev_ut_hit_offsets>(),
    //   arguments.offset<arg::dev_atomics_storage>(),
    //   arguments.offset<arg::dev_velo_track_hit_number>(),
    //   arguments.offset<arg::dev_velo_track_hits>(),
    //   arguments.offset<arg::dev_velo_states>(),
    //   arguments.offset<arg::dev_veloUT_tracks>(),
    //   arguments.offset<arg::dev_atomics_veloUT>(),
    //   dev_ut_magnet_tool,
    //   constants.dev_ut_dxDy,
    //   constants.dev_unique_x_sector_layer_offsets,
    //   constants.dev_unique_x_sector_offsets,
    //   constants.dev_unique_sector_xs
    // );
    // sequence.invoke<seq::veloUT>();

    // // Transmission device to host
    // // Velo tracks
    // cudaCheck(cudaMemcpyAsync(host_velo_tracks_atomics, arguments.offset<arg::dev_atomics_storage>(), (2 * number_of_events + 1) * sizeof(uint), cudaMemcpyDeviceToHost, stream));
    // cudaCheck(cudaMemcpyAsync(host_velo_track_hit_number, arguments.offset<arg::dev_velo_track_hit_number>(), arguments.size<arg::dev_velo_track_hit_number>(), cudaMemcpyDeviceToHost, stream));
    // cudaCheck(cudaMemcpyAsync(host_velo_track_hits, arguments.offset<arg::dev_velo_track_hits>(), host_accumulated_number_of_hits_in_velo_tracks[0] * sizeof(Velo::Hit), cudaMemcpyDeviceToHost, stream));
    // cudaCheck(cudaMemcpyAsync(host_velo_states, arguments.offset<arg::dev_velo_states>(), host_number_of_reconstructed_velo_tracks[0] * sizeof(Velo::State), cudaMemcpyDeviceToHost, stream));

    // // VeloUT tracks
    // cudaCheck(cudaMemcpyAsync(host_atomics_veloUT, arguments.offset<arg::dev_atomics_veloUT>(), arguments.size<arg::dev_atomics_veloUT>(), cudaMemcpyDeviceToHost, stream));
    // cudaCheck(cudaMemcpyAsync(host_veloUT_tracks, arguments.offset<arg::dev_veloUT_tracks>(), arguments.size<arg::dev_veloUT_tracks>(), cudaMemcpyDeviceToHost, stream));

    // // SciFi preprocessing
    // // Estimate cluster count
    // arguments.set_size<arg::dev_scifi_raw_input>(host_scifi_events_size);
    // arguments.set_size<arg::dev_scifi_raw_input_offsets>(host_scifi_event_offsets_size);
    // arguments.set_size<arg::dev_scifi_hit_count>(2 * number_of_events * SciFi::number_of_zones + 1);

    // scheduler.setup_next(arguments, sequence_step++);

    // cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_scifi_raw_input>(), host_scifi_events, host_scifi_events_size, cudaMemcpyHostToDevice, stream));
    // cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_scifi_raw_input_offsets>(), host_scifi_event_offsets, host_scifi_event_offsets_size * sizeof(uint), cudaMemcpyHostToDevice, stream));
    // cudaCheck(cudaMemsetAsync(arguments.offset<arg::dev_scifi_hit_count>(), 0, arguments.size<arg::dev_scifi_hit_count>(), stream));
    // cudaEventRecord(cuda_generic_event, stream);
    // cudaEventSynchronize(cuda_generic_event);

    // sequence.set_opts<seq::estimate_cluster_count>(dim3(number_of_events), dim3(240), stream);
    // sequence.set_arguments<seq::estimate_cluster_count>(
    //   arguments.offset<arg::dev_scifi_raw_input>(),
    //   arguments.offset<arg::dev_scifi_raw_input_offsets>(),
    //   arguments.offset<arg::dev_scifi_hit_count>(),
    //   dev_scifi_geometry
    // );
    // sequence.invoke<seq::estimate_cluster_count>();

    // cudaEventRecord(cuda_generic_event, stream);
    // cudaEventSynchronize(cuda_generic_event);

    // // Prefix sum of hit count (becomes hit offset)
    // // 1. Reduce
    // // 2. Single block
    // // 3. Scan

    // // Prefix sum: Reduce
    // const uint total_number_of_zones = number_of_events * SciFi::number_of_zones;
    // const size_t prefix_sum_auxiliary_array_4_size = (total_number_of_zones + 511) / 512;
    // arguments.set_size<arg::dev_prefix_sum_auxiliary_array_4>(prefix_sum_auxiliary_array_4_size);
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_opts<seq::prefix_sum_reduce_scifi_hits>(dim3(prefix_sum_auxiliary_array_4_size), dim3(256), stream);
    // sequence.set_arguments<seq::prefix_sum_reduce_scifi_hits>(
    //   arguments.offset<arg::dev_scifi_hit_count>(),
    //   arguments.offset<arg::dev_prefix_sum_auxiliary_array_4>(),
    //   total_number_of_zones
    // );
    // sequence.invoke<seq::prefix_sum_reduce_scifi_hits>();

    // // Prefix sum: Single block
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_arguments<seq::prefix_sum_single_block_scifi_hits>(
    //   arguments.offset<arg::dev_scifi_hit_count>() + total_number_of_zones,
    //   arguments.offset<arg::dev_prefix_sum_auxiliary_array_4>(),
    //   prefix_sum_auxiliary_array_4_size
    // );
    // sequence.invoke<seq::prefix_sum_single_block_scifi_hits>();

    // // Prefix sum: Scan
    // scheduler.setup_next(arguments, sequence_step++);
    // const uint pss_scifi_hits_blocks = prefix_sum_auxiliary_array_4_size==1 ? 1 : (prefix_sum_auxiliary_array_4_size-1);
    // sequence.set_opts<seq::prefix_sum_scan_scifi_hits>(dim3(pss_scifi_hits_blocks), dim3(512), stream);
    // sequence.set_arguments<seq::prefix_sum_scan_scifi_hits>(
    //   arguments.offset<arg::dev_scifi_hit_count>(),
    //   arguments.offset<arg::dev_prefix_sum_auxiliary_array_4>(),
    //   total_number_of_zones
    // );
    // sequence.invoke<seq::prefix_sum_scan_scifi_hits>();

    // // Fetch total number of hits
    // cudaCheck(cudaMemcpyAsync(host_accumulated_number_of_scifi_hits,
    //   arguments.offset<arg::dev_scifi_hit_count>() + total_number_of_zones,
    //   sizeof(uint), cudaMemcpyDeviceToHost, stream));
    // cudaEventRecord(cuda_generic_event, stream);
    // cudaEventSynchronize(cuda_generic_event);

    // // info_cout << "Total SciFi cluster estimate: " << *host_accumulated_number_of_scifi_hits << std::endl;


    // // Raw Bank Decoder
    // const uint32_t hits_bytes = (14 * sizeof(float) + 1) * *host_accumulated_number_of_scifi_hits;
    // arguments.set_size<arg::dev_scifi_hits>(hits_bytes);

    // scheduler.setup_next(arguments, sequence_step++);

    // sequence.set_opts<seq::raw_bank_decoder>(dim3(number_of_events), dim3(240), stream);
    // sequence.set_arguments<seq::raw_bank_decoder>(
    //   arguments.offset<arg::dev_scifi_raw_input>(),
    //   arguments.offset<arg::dev_scifi_raw_input_offsets>(),
    //   arguments.offset<arg::dev_scifi_hit_count>(),
    //   arguments.offset<arg::dev_scifi_hits>(),
    //   dev_scifi_geometry
    // );

    // sequence.invoke<seq::raw_bank_decoder>();

    // cudaEventRecord(cuda_generic_event, stream);
    // cudaEventSynchronize(cuda_generic_event);

    // // SciFi hit sorting by x
    // arguments.set_size<arg::dev_scifi_hit_permutations>(*host_accumulated_number_of_scifi_hits);
    // scheduler.setup_next(arguments, sequence_step++);
    // sequence.set_opts<seq::scifi_sort_by_x>(dim3(number_of_events), dim3(64), stream);
    // sequence.set_arguments<seq::scifi_sort_by_x>(
    //   arguments.offset<arg::dev_scifi_hits>(),
    //   arguments.offset<arg::dev_scifi_hit_count>(),
    //   arguments.offset<arg::dev_scifi_hit_permutations>()
    // );
    // sequence.invoke<seq::scifi_sort_by_x>();

    // /*
    // // SciFi Decoder Debugging
    // const uint hit_count_uints = 2 * number_of_events * SciFi::number_of_zones + 1;
    // uint host_scifi_hit_count[hit_count_uints];
    // char* host_scifi_hits = new char[hits_bytes];
    // uint* host_scifi_hit_permutation = new uint[*host_accumulated_number_of_scifi_hits];
    // cudaCheck(cudaMemcpyAsync(&host_scifi_hit_count, arguments.offset<arg::dev_scifi_hit_count>(), hit_count_uints*sizeof(uint), cudaMemcpyDeviceToHost, stream));
    // cudaCheck(cudaMemcpyAsync(host_scifi_hits, arguments.offset<arg::dev_scifi_hits>(), arguments.size<hits_bytes>(), cudaMemcpyDeviceToHost, stream));
    // cudaCheck(cudaMemcpyAsync(host_scifi_hit_permutation, arguments.offset<arg::dev_scifi_hit_permutations>(), arguments.offset<arg::dev_scifi_hit_permutations>(), cudaMemcpyDeviceToHost, stream));
    // cudaEventRecord(cuda_generic_event, stream);
    // cudaEventSynchronize(cuda_generic_event);

    // SciFi::SciFiHits host_scifi_hits_struct;
    // host_scifi_hits_struct.typecast_sorted(host_scifi_hits, host_scifi_hit_count[number_of_events * SciFi::number_of_zones]);

    // //Print only non-empty hits
    // std::ofstream outfile("dump.txt");
    // SciFi::SciFiHitCount host_scifi_hit_count_struct;
    // for(size_t event = 0; event < number_of_events; event++) {
    //   host_scifi_hit_count_struct.typecast_ascifier_prefix_sum(host_scifi_hit_count, event, number_of_events);
    //   for(size_t zone = 0; zone < SciFi::number_of_zones; zone++) {
    //     for(size_t hit = 0; hit < host_scifi_hit_count_struct.n_hits_layers[zone]; hit++) {
    //       auto h = host_scifi_hits_struct.getHit(host_scifi_hit_count_struct.layer_offsets[zone] + hit);
    //       outfile << std::setprecision(8) << std::fixed << h.planeCode << " " << h.hitZone << " " << h.LHCbID << " "
    //         << h.x0 << " " << h.z0 << " " << h.w<< " " << h.dxdy << " "
    //         << h.dzdy << " " << h.yMin << " " << h.yMax  <<  std::endl;
    //     }
    //   }
    // }*/

    // ///////////////////////
    // // Monte Carlo Check //
    // ///////////////////////

    // if (do_check && i_stream == 0) {
    //   if (repetition == 0) { // only check efficiencies once
    //     std::cout << "Checking Velo tracks reconstructed on GPU" << std::endl;

    //     const std::vector<trackChecker::Tracks> tracks_events = prepareTracks(
    //       host_velo_tracks_atomics,
    //       host_velo_track_hit_number,
    //       host_velo_track_hits,
    //       number_of_events);

    //     std::string trackType = "Velo";
    //     call_pr_checker(
    //       tracks_events,
    //       folder_name_MC,
    //       start_event_offset,
    //       trackType
    //     );

    //     /* CHECKING VeloUT TRACKS */
    //     const std::vector< trackChecker::Tracks > veloUT_tracks = prepareVeloUTTracks(
    //       host_veloUT_tracks,
    //       host_atomics_veloUT,
    //       number_of_events
    //     );

    //     std::cout << "Checking VeloUT tracks reconstructed on GPU" << std::endl;
    //     trackType = "VeloUT";
    //     call_pr_checker (
    //       veloUT_tracks,
    //       folder_name_MC,
    //       start_event_offset,
    //       trackType
    //     );
    //   } // only in first repetition
    // } // do_check
  } // repetitions

  return cudaSuccess;
}
