// #include "Stream.cuh"
// #include <cuda_profiler_api.h>

// #include <iostream>
// #include <fstream>
// #include <iomanip>

// // run those algorithms that have an x86 implementation on x86
// // run all others on the GPU
// // copy necessary input for x86 algorithms from device to host
// cudaError_t Stream::run_sequence_on_x86(
//   const uint i_stream,
//   const char* host_velopix_events,
//   const uint* host_velopix_event_offsets,
//   const size_t host_velopix_events_size,
//   const size_t host_velopix_event_offsets_size,
//   const char* host_ut_events,
//   const uint* host_ut_event_offsets,
//   const size_t host_ut_events_size,
//   const size_t host_ut_event_offsets_size,
//   const char* host_scifi_events,
//   const uint* host_scifi_event_offsets,
//   const size_t host_scifi_events_size,
//   const size_t host_scifi_event_offsets_size,
//   const uint number_of_events,
//   const uint number_of_repetitions
// ) {
//   // Generate object for populating arguments
//   ArgumentManager<argument_tuple_t> arguments {dev_base_pointer};

//   for (uint repetition=0; repetition<number_of_repetitions; ++repetition) {
//     uint sequence_step = 0;

//     cudaProfilerStart(); 

//     // Reset scheduler
//     scheduler.reset();

//     // Estimate input size
//     // Set arguments and reserve memory
//     arguments.set_size<arg::dev_raw_input>(host_velopix_events_size);
//     arguments.set_size<arg::dev_raw_input_offsets>(host_velopix_event_offsets_size);
//     arguments.set_size<arg::dev_estimated_input_size>(number_of_events * VeloTracking::n_modules + 1);
//     arguments.set_size<arg::dev_module_cluster_num>(number_of_events * VeloTracking::n_modules);
//     arguments.set_size<arg::dev_module_candidate_num>(number_of_events);
//     arguments.set_size<arg::dev_cluster_candidates>(number_of_events * VeloClustering::max_candidates_event);
//     scheduler.setup_next(arguments, sequence_step++);
//     // Setup opts and arguments for kernel call
//     sequence.set_opts<seq::estimate_input_size>(dim3(number_of_events), dim3(32, 26), stream);
//     sequence.set_arguments<seq::estimate_input_size>(
//       arguments.offset<arg::dev_raw_input>(),
//       arguments.offset<arg::dev_raw_input_offsets>(),
//       arguments.offset<arg::dev_estimated_input_size>(),
//       arguments.offset<arg::dev_module_cluster_num>(),
//       arguments.offset<arg::dev_module_candidate_num>(),
//       arguments.offset<arg::dev_cluster_candidates>(),
//       constants.dev_velo_candidate_ks
//     );

//     cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_raw_input>(), host_velopix_events, arguments.size<arg::dev_raw_input>(), cudaMemcpyHostToDevice, stream));
//     cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_raw_input_offsets>(), host_velopix_event_offsets, arguments.size<arg::dev_raw_input_offsets>(), cudaMemcpyHostToDevice, stream));
//     cudaEventRecord(cuda_generic_event, stream);
//     cudaEventSynchronize(cuda_generic_event);

//     // Kernel call
//     sequence.invoke<seq::estimate_input_size>();

//     // Convert the estimated sizes to module hit start format (argument_offsets)
//     // Set arguments and reserve memory
//     arguments.set_size<arg::dev_cluster_offset>(number_of_events);
//     scheduler.setup_next(arguments, sequence_step++);
//     // Setup sequence step
//     const auto prefix_sum_blocks = (VeloTracking::n_modules * number_of_events + 511) / 512;
//     sequence.set_opts<seq::prefix_sum_reduce>(dim3(prefix_sum_blocks), dim3(256), stream);
//     sequence.set_arguments<seq::prefix_sum_reduce>(
//       arguments.offset<arg::dev_estimated_input_size>(),
//       arguments.offset<arg::dev_cluster_offset>(),
//       VeloTracking::n_modules * number_of_events
//     );
//     // Kernel call
//     sequence.invoke<seq::prefix_sum_reduce>();

//     // Prefix Sum Single Block
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_arguments<seq::prefix_sum_single_block>(
//       arguments.offset<arg::dev_estimated_input_size>() + VeloTracking::n_modules * number_of_events,
//       arguments.offset<arg::dev_cluster_offset>(),
//       prefix_sum_blocks
//     );
//     sequence.invoke<seq::prefix_sum_single_block>();

//     // Prefix sum scan
//     scheduler.setup_next(arguments, sequence_step++);
//     const auto prefix_sum_scan_blocks = prefix_sum_blocks==1 ? 1 : (prefix_sum_blocks-1);
//     sequence.set_opts<seq::prefix_sum_scan>(dim3(prefix_sum_scan_blocks), dim3(512), stream);
//     sequence.set_arguments<seq::prefix_sum_scan>(
//       arguments.offset<arg::dev_estimated_input_size>(),
//       arguments.offset<arg::dev_cluster_offset>(),
//       VeloTracking::n_modules * number_of_events
//     );
//     sequence.invoke<seq::prefix_sum_scan>();

//     // Fetch the number of hits we require
//     cudaCheck(cudaMemcpyAsync(host_total_number_of_velo_clusters, arguments.offset<arg::dev_estimated_input_size>() + number_of_events * VeloTracking::n_modules, sizeof(uint), cudaMemcpyDeviceToHost, stream));
//     cudaEventRecord(cuda_generic_event, stream);
//     cudaEventSynchronize(cuda_generic_event);

//     // Masked Velo clustering
//     arguments.set_size<arg::dev_velo_cluster_container>(6 * host_total_number_of_velo_clusters[0]);
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_opts<seq::masked_velo_clustering>(dim3(number_of_events), dim3(256), stream);
//     sequence.set_arguments<seq::masked_velo_clustering>(
//       arguments.offset<arg::dev_raw_input>(),
//       arguments.offset<arg::dev_raw_input_offsets>(),
//       arguments.offset<arg::dev_estimated_input_size>(),
//       arguments.offset<arg::dev_module_cluster_num>(),
//       arguments.offset<arg::dev_module_candidate_num>(),
//       arguments.offset<arg::dev_cluster_candidates>(),
//       arguments.offset<arg::dev_velo_cluster_container>(),
//       dev_velo_geometry,
//       constants.dev_velo_sp_patterns,
//       constants.dev_velo_sp_fx,
//       constants.dev_velo_sp_fy
//     );
//     sequence.invoke<seq::masked_velo_clustering>();

//     // Calculate phi and sort
//     arguments.set_size<arg::dev_hit_permutation>(host_total_number_of_velo_clusters[0]);
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_opts<seq::calculate_phi_and_sort>(dim3(number_of_events), dim3(64), stream);
//     sequence.set_arguments<seq::calculate_phi_and_sort>(
//       arguments.offset<arg::dev_estimated_input_size>(),
//       arguments.offset<arg::dev_module_cluster_num>(),
//       arguments.offset<arg::dev_velo_cluster_container>(),
//       arguments.offset<arg::dev_hit_permutation>()
//     );
//     sequence.invoke<seq::calculate_phi_and_sort>();

//     // Fill candidates
//     arguments.set_size<arg::dev_h0_candidates>(2 * host_total_number_of_velo_clusters[0]);
//     arguments.set_size<arg::dev_h2_candidates>(2 * host_total_number_of_velo_clusters[0]);
//     scheduler.setup_next(arguments, sequence_step++);
//     // Setup opts and arguments
//     sequence.set_opts<seq::fill_candidates>(dim3(number_of_events, 48), dim3(128), stream);
//     sequence.set_arguments<seq::fill_candidates>(
//       arguments.offset<arg::dev_velo_cluster_container>(),
//       arguments.offset<arg::dev_estimated_input_size>(),
//       arguments.offset<arg::dev_module_cluster_num>(),
//       arguments.offset<arg::dev_h0_candidates>(),
//       arguments.offset<arg::dev_h2_candidates>()
//     );
//     sequence.invoke<seq::fill_candidates>();

//     // Search by triplet
//     arguments.set_size<arg::dev_tracks>(number_of_events * VeloTracking::max_tracks);
//     arguments.set_size<arg::dev_tracklets>(number_of_events * VeloTracking::ttf_modulo);
//     arguments.set_size<arg::dev_tracks_to_follow>(number_of_events * VeloTracking::ttf_modulo);
//     arguments.set_size<arg::dev_weak_tracks>(number_of_events * VeloTracking::max_weak_tracks);
//     arguments.set_size<arg::dev_hit_used>(host_total_number_of_velo_clusters[0]);
//     arguments.set_size<arg::dev_atomics_storage>(number_of_events * VeloTracking::num_atomics);
//     arguments.set_size<arg::dev_rel_indices>(number_of_events * 2 * VeloTracking::max_numhits_in_module);
//     scheduler.setup_next(arguments, sequence_step++);
//     // Setup opts and arguments
//     sequence.set_opts<seq::search_by_triplet>(dim3(number_of_events), dim3(32), stream, 32 * sizeof(float));
//     sequence.set_arguments<seq::search_by_triplet>(
//       arguments.offset<arg::dev_velo_cluster_container>(),
//       arguments.offset<arg::dev_estimated_input_size>(),
//       arguments.offset<arg::dev_module_cluster_num>(),
//       arguments.offset<arg::dev_tracks>(),
//       arguments.offset<arg::dev_tracklets>(),
//       arguments.offset<arg::dev_tracks_to_follow>(),
//       arguments.offset<arg::dev_weak_tracks>(),
//       arguments.offset<arg::dev_hit_used>(),
//       arguments.offset<arg::dev_atomics_storage>(),
//       arguments.offset<arg::dev_h0_candidates>(),
//       arguments.offset<arg::dev_h2_candidates>(),
//       arguments.offset<arg::dev_rel_indices>(),
//       constants.dev_velo_module_zs
//     );
//     sequence.invoke<seq::search_by_triplet>();

//     // Weak tracks adder
//     scheduler.setup_next(arguments, sequence_step++);
//     // Setup opts and arguments
//     sequence.set_opts<seq::weak_tracks_adder>(dim3(number_of_events), dim3(256), stream);
//     sequence.set_arguments<seq::weak_tracks_adder>(
//       arguments.offset<arg::dev_velo_cluster_container>(),
//       arguments.offset<arg::dev_estimated_input_size>(),
//       arguments.offset<arg::dev_tracks>(),
//       arguments.offset<arg::dev_weak_tracks>(),
//       arguments.offset<arg::dev_hit_used>(),
//       arguments.offset<arg::dev_atomics_storage>()
//     );
//     sequence.invoke<seq::weak_tracks_adder>();

//     // Calculate prefix sum of found tracks
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_arguments<seq::copy_and_prefix_sum_single_block>(
//       (uint*) arguments.offset<arg::dev_atomics_storage>() + number_of_events*2,
//       (uint*) arguments.offset<arg::dev_atomics_storage>(),
//       (uint*) arguments.offset<arg::dev_atomics_storage>() + number_of_events,
//       number_of_events
//     );
//     sequence.invoke<seq::copy_and_prefix_sum_single_block>();

//     // Fetch number of reconstructed tracks
//     cudaCheck(cudaMemcpyAsync(host_number_of_reconstructed_velo_tracks, arguments.offset<arg::dev_atomics_storage>() + number_of_events * 2, sizeof(uint), cudaMemcpyDeviceToHost, stream));
//     cudaEventRecord(cuda_generic_event, stream);
//     cudaEventSynchronize(cuda_generic_event);
//     //host_number_of_reconstructed_velo_tracks[0] = 0;
//     size_t velo_track_hit_number_size = host_number_of_reconstructed_velo_tracks[0] + 1;
    
//     // Prefix sum of tracks hits
//     // 1. Copy velo track hit number to a consecutive container
//     // 2. Reduce
//     // 3. Single block
//     // 4. Scan

//     // Copy Velo track hit number
//     arguments.set_size<arg::dev_velo_track_hit_number>(velo_track_hit_number_size);
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_opts<seq::copy_velo_track_hit_number>(dim3(number_of_events), dim3(512), stream);
//     sequence.set_arguments<seq::copy_velo_track_hit_number>(
//       arguments.offset<arg::dev_tracks>(),
//       arguments.offset<arg::dev_atomics_storage>(),
//       arguments.offset<arg::dev_velo_track_hit_number>()
//     );
//     sequence.invoke<seq::copy_velo_track_hit_number>();

//     // Prefix sum: Reduce
//     const size_t prefix_sum_auxiliary_array_2_size = (host_number_of_reconstructed_velo_tracks[0] + 511) / 512;
//     arguments.set_size<arg::dev_prefix_sum_auxiliary_array_2>(prefix_sum_auxiliary_array_2_size);
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_opts<seq::prefix_sum_reduce_velo_track_hit_number>(dim3(prefix_sum_auxiliary_array_2_size), dim3(256), stream);
//     sequence.set_arguments<seq::prefix_sum_reduce_velo_track_hit_number>(
//       arguments.offset<arg::dev_velo_track_hit_number>(),
//       arguments.offset<arg::dev_prefix_sum_auxiliary_array_2>(),
//       host_number_of_reconstructed_velo_tracks[0]
//     );
//     sequence.invoke<seq::prefix_sum_reduce_velo_track_hit_number>();

//     // Prefix sum: Single block
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_arguments<seq::prefix_sum_single_block_velo_track_hit_number>(
//       arguments.offset<arg::dev_velo_track_hit_number>() + host_number_of_reconstructed_velo_tracks[0],
//       arguments.offset<arg::dev_prefix_sum_auxiliary_array_2>(),
//       prefix_sum_auxiliary_array_2_size
//     );
//     sequence.invoke<seq::prefix_sum_single_block_velo_track_hit_number>();

//     // Prefix sum: Scan
//     scheduler.setup_next(arguments, sequence_step++);
//     const uint pss_velo_track_hit_number_opts =
//       prefix_sum_auxiliary_array_2_size==1 ? 1 : (prefix_sum_auxiliary_array_2_size-1);
//     sequence.set_opts<seq::prefix_sum_scan_velo_track_hit_number>(dim3(pss_velo_track_hit_number_opts), dim3(512), stream);
//     sequence.set_arguments<seq::prefix_sum_scan_velo_track_hit_number>(
//       arguments.offset<arg::dev_velo_track_hit_number>(),
//       arguments.offset<arg::dev_prefix_sum_auxiliary_array_2>(),
//       host_number_of_reconstructed_velo_tracks[0]
//     );
//     sequence.invoke<seq::prefix_sum_scan_velo_track_hit_number>();

//     // Fetch total number of hits accumulated with all tracks
//     cudaCheck(cudaMemcpyAsync(host_accumulated_number_of_hits_in_velo_tracks,
//       arguments.offset<arg::dev_velo_track_hit_number>() + host_number_of_reconstructed_velo_tracks[0],
//       sizeof(uint), cudaMemcpyDeviceToHost, stream));
//     cudaEventRecord(cuda_generic_event, stream);
//     cudaEventSynchronize(cuda_generic_event);
//     // host_accumulated_number_of_hits_in_velo_tracks[0] = 0;
//     // host_number_of_reconstructed_velo_tracks[0] = 0;

//     // Consolidate tracks
//     // TODO: The size specified (sizeof(Hits) / sizeof(uint)) is due to the
//     //       lgenfe error from the nvcc compiler, present in Cuda 9.2. Once it
//     //       is gone, we can switch all pointers to char*.
//     arguments.set_size<arg::dev_velo_track_hits>(host_accumulated_number_of_hits_in_velo_tracks[0] * sizeof(Velo::Hit) / sizeof(uint));
//     arguments.set_size<arg::dev_velo_states>(host_number_of_reconstructed_velo_tracks[0] * sizeof(Velo::State) / sizeof(uint));
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_opts<seq::consolidate_tracks>(dim3(number_of_events), dim3(256), stream);
//     sequence.set_arguments<seq::consolidate_tracks>(
//       arguments.offset<arg::dev_atomics_storage>(),
//       arguments.offset<arg::dev_tracks>(),
//       arguments.offset<arg::dev_velo_track_hit_number>(),
//       arguments.offset<arg::dev_velo_cluster_container>(),
//       arguments.offset<arg::dev_estimated_input_size>(),
//       arguments.offset<arg::dev_module_cluster_num>(),
//       arguments.offset<arg::dev_velo_track_hits>(),
//       arguments.offset<arg::dev_velo_states>()
//     );
//     sequence.invoke<seq::consolidate_tracks>();

//     // Calculate number of UT hits
//     // Set arguments and reserve memory
//     arguments.set_size<arg::dev_ut_raw_input>(host_ut_events_size);
//     arguments.set_size<arg::dev_ut_raw_input_offsets>(host_ut_event_offsets_size);
//     arguments.set_size<arg::dev_ut_hit_offsets>(number_of_events * constants.host_unique_x_sector_layer_offsets[4] + 1);
//     scheduler.setup_next(arguments, sequence_step++);
//     // Setup opts and arguments for kernel call
//     cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_ut_raw_input>(), host_ut_events, host_ut_events_size, cudaMemcpyHostToDevice, stream));
//     cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_ut_raw_input_offsets>(), host_ut_event_offsets, host_ut_event_offsets_size * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
//     cudaCheck(cudaMemsetAsync(arguments.offset<arg::dev_ut_hit_offsets>(), 0, arguments.size<arg::dev_ut_hit_offsets>(), stream));
//     sequence.set_opts<seq::ut_calculate_number_of_hits>(dim3(number_of_events), dim3(64, 4), stream);
//     sequence.set_arguments<seq::ut_calculate_number_of_hits>(
//       arguments.offset<arg::dev_ut_raw_input>(),
//       arguments.offset<arg::dev_ut_raw_input_offsets>(),
//       dev_ut_boards,
//       constants.dev_ut_region_offsets,
//       constants.dev_unique_x_sector_layer_offsets,
//       constants.dev_unique_x_sector_offsets,
//       arguments.offset<arg::dev_ut_hit_offsets>()
//     );
//     // Invoke kernel
//     sequence.invoke<seq::ut_calculate_number_of_hits>();

//     // // Print UT hit count per event per layer
//     // std::vector<uint> host_ut_hit_count (number_of_events * constants.host_unique_x_sector_layer_offsets[4] + 1);
//     // cudaCheck(cudaMemcpyAsync(host_ut_hit_count.data(), arguments.offset<arg::dev_ut_hit_offsets>(), argen.size<arg::dev_ut_hit_offsets>(number_of_events * constants.host_unique_x_sector_layer_offsets[4] + 1), cudaMemcpyDeviceToHost, stream));
//     // cudaEventRecord(cuda_generic_event, stream);
//     // cudaEventSynchronize(cuda_generic_event);
//     // for (int e=0; e<number_of_events; ++e) {
//     //   info_cout << "Event " << e << ", #hits per layer: ";
//     //   uint32_t* count = host_ut_hit_count.data() + e * constants.host_unique_x_sector_layer_offsets[4];
//     //   for (uint32_t i = 0; i < constants.host_unique_x_sector_layer_offsets[4]; ++i) {
//     //     info_cout << count[i] << ", ";
//     //   }
//     //   info_cout << std::endl;
//     // }

//     // Prefix sum of hit count (becomes hit offset)
//     // 1. Reduce
//     // 2. Single block
//     // 3. Scan

//     // Prefix sum: Reduce
//     const uint total_number_of_sectors = number_of_events * constants.host_unique_x_sector_layer_offsets[4];
//     const size_t prefix_sum_auxiliary_array_3_size = (total_number_of_sectors + 511) / 512;
//     arguments.set_size<arg::dev_prefix_sum_auxiliary_array_3>(prefix_sum_auxiliary_array_3_size);
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_opts<seq::prefix_sum_reduce_ut_hits>(dim3(prefix_sum_auxiliary_array_3_size), dim3(256), stream);
//     sequence.set_arguments<seq::prefix_sum_reduce_ut_hits>(
//       arguments.offset<arg::dev_ut_hit_offsets>(),
//       arguments.offset<arg::dev_prefix_sum_auxiliary_array_3>(),
//       total_number_of_sectors
//     );
//     sequence.invoke<seq::prefix_sum_reduce_ut_hits>();

//     // Prefix sum: Single block
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_arguments<seq::prefix_sum_single_block_ut_hits>(
//       arguments.offset<arg::dev_ut_hit_offsets>() + total_number_of_sectors,
//       arguments.offset<arg::dev_prefix_sum_auxiliary_array_3>(),
//       prefix_sum_auxiliary_array_3_size
//     );
//     sequence.invoke<seq::prefix_sum_single_block_ut_hits>();

//     // Prefix sum: Scan
//     scheduler.setup_next(arguments, sequence_step++);
//     const uint pss_ut_hits_blocks = prefix_sum_auxiliary_array_3_size==1 ? 1 : (prefix_sum_auxiliary_array_3_size-1);
//     sequence.set_opts<seq::prefix_sum_scan_ut_hits>(dim3(pss_ut_hits_blocks), dim3(512), stream);
//     sequence.set_arguments<seq::prefix_sum_scan_ut_hits>(
//       arguments.offset<arg::dev_ut_hit_offsets>(),
//       arguments.offset<arg::dev_prefix_sum_auxiliary_array_3>(),
//       total_number_of_sectors
//     );
//     sequence.invoke<seq::prefix_sum_scan_ut_hits>();

//     // Fetch total number of hits accumulated with all tracks
//     cudaCheck(cudaMemcpyAsync(host_accumulated_number_of_ut_hits,
//       arguments.offset<arg::dev_ut_hit_offsets>() + total_number_of_sectors,
//       sizeof(uint), cudaMemcpyDeviceToHost, stream));
//     cudaEventRecord(cuda_generic_event, stream);
//     cudaEventSynchronize(cuda_generic_event);

//     // // Now, we should have the offset instead, and the sum of all in host_accumulated_number_of_ut_hits
//     // // Check that
//     // cudaCheck(cudaMemcpyAsync(host_ut_hit_count.data(), arguments.offset<arg::dev_ut_hit_offsets>(), arguments.size<arg::dev_ut_hit_offsets>(), cudaMemcpyDeviceToHost, stream));
//     // cudaEventRecord(cuda_generic_event, stream);
//     // cudaEventSynchronize(cuda_generic_event);
//     // for (int e=0; e<number_of_events; ++e) {
//     //   info_cout << "Event " << e << ", offset per sector group: ";
//     //   uint32_t* offset = host_ut_hit_count.data() + e * constants.host_unique_x_sector_layer_offsets[4];
//     //   for (uint32_t i = 0; i < constants.host_unique_x_sector_layer_offsets[4]; ++i) {
//     //     info_cout << offset[i] << ", ";
//     //   }
//     //   info_cout << std::endl;
//     // }
//     // info_cout << "Total number of UT hits: " << *host_accumulated_number_of_ut_hits << std::endl;

//     // UT pre-decoding
//     arguments.set_size<arg::dev_ut_hits>(UTHits::number_of_arrays * host_accumulated_number_of_ut_hits[0]);
//     arguments.set_size<arg::dev_ut_hit_count>(number_of_events * constants.host_unique_x_sector_layer_offsets[4]);
//     scheduler.setup_next(arguments, sequence_step++);
//     cudaCheck(cudaMemsetAsync(arguments.offset<arg::dev_ut_hit_count>(), 0, arguments.size<arg::dev_ut_hit_count>(), stream));
//     sequence.set_opts<seq::ut_pre_decode>(dim3(number_of_events), dim3(64, 4), stream);
//     sequence.set_arguments<seq::ut_pre_decode>(
//       arguments.offset<arg::dev_ut_raw_input>(),
//       arguments.offset<arg::dev_ut_raw_input_offsets>(),
//       dev_ut_boards,
//       dev_ut_geometry,
//       constants.dev_ut_region_offsets,
//       constants.dev_unique_x_sector_layer_offsets,
//       constants.dev_unique_x_sector_offsets,
//       arguments.offset<arg::dev_ut_hit_offsets>(),
//       arguments.offset<arg::dev_ut_hits>(),
//       arguments.offset<arg::dev_ut_hit_count>()
//     );
//     sequence.invoke<seq::ut_pre_decode>();
    
//     // UT find permutation by looking at y
//     arguments.set_size<arg::dev_ut_hit_permutations>(host_accumulated_number_of_ut_hits[0]);
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_opts<seq::ut_find_permutation>(dim3(number_of_events, constants.host_unique_x_sector_layer_offsets[4]), dim3(16), stream);
//     sequence.set_arguments<seq::ut_find_permutation>(
//       arguments.offset<arg::dev_ut_hits>(),
//       arguments.offset<arg::dev_ut_hit_offsets>(),
//       arguments.offset<arg::dev_ut_hit_permutations>(),
//       constants.dev_unique_x_sector_layer_offsets,
//       constants.dev_unique_x_sector_offsets,
//       constants.dev_unique_sector_xs
//     );
//     sequence.invoke<seq::ut_find_permutation>();

//     // UT decode sorted
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_opts<seq::ut_decode_raw_banks_in_order>(dim3(number_of_events, VeloUTTracking::n_layers), dim3(64), stream);
//     sequence.set_arguments<seq::ut_decode_raw_banks_in_order>(
//       arguments.offset<arg::dev_ut_raw_input>(),
//       arguments.offset<arg::dev_ut_raw_input_offsets>(),
//       dev_ut_boards,
//       dev_ut_geometry,
//       constants.dev_ut_region_offsets,
//       constants.dev_unique_x_sector_layer_offsets,
//       constants.dev_unique_x_sector_offsets,
//       arguments.offset<arg::dev_ut_hit_offsets>(),
//       arguments.offset<arg::dev_ut_hits>(),
//       arguments.offset<arg::dev_ut_hit_count>(),
//       arguments.offset<arg::dev_ut_hit_permutations>()
//     );
//     sequence.invoke<seq::ut_decode_raw_banks_in_order>();
    
//     // VeloUT tracking
//     arguments.set_size<arg::dev_veloUT_tracks>(number_of_events * VeloUTTracking::max_num_tracks);
//     arguments.set_size<arg::dev_atomics_veloUT>(VeloUTTracking::num_atomics * number_of_events);
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_opts<seq::veloUT>(dim3(number_of_events), dim3(32), stream);
//     sequence.set_arguments<seq::veloUT>(
//       arguments.offset<arg::dev_ut_hits>(),
//       arguments.offset<arg::dev_ut_hit_offsets>(),
//       arguments.offset<arg::dev_atomics_storage>(),
//       arguments.offset<arg::dev_velo_track_hit_number>(),
//       arguments.offset<arg::dev_velo_track_hits>(),
//       arguments.offset<arg::dev_velo_states>(),
//       arguments.offset<arg::dev_veloUT_tracks>(),
//       arguments.offset<arg::dev_atomics_veloUT>(),
//       dev_ut_magnet_tool,
//       constants.dev_ut_dxDy,
//       constants.dev_unique_x_sector_layer_offsets,
//       constants.dev_unique_x_sector_offsets,
//       constants.dev_unique_sector_xs
//     );
//     sequence.invoke<seq::veloUT>();

//     // Transmission device to host
//     // Velo tracks
//     cudaCheck(cudaMemcpyAsync(host_velo_tracks_atomics, arguments.offset<arg::dev_atomics_storage>(), (2 * number_of_events + 1) * sizeof(uint), cudaMemcpyDeviceToHost, stream));
//     cudaCheck(cudaMemcpyAsync(host_velo_track_hit_number, arguments.offset<arg::dev_velo_track_hit_number>(), arguments.size<arg::dev_velo_track_hit_number>(), cudaMemcpyDeviceToHost, stream));
//     cudaCheck(cudaMemcpyAsync(host_velo_track_hits, arguments.offset<arg::dev_velo_track_hits>(), host_accumulated_number_of_hits_in_velo_tracks[0] * sizeof(Velo::Hit), cudaMemcpyDeviceToHost, stream));
//     cudaCheck(cudaMemcpyAsync(host_velo_states, arguments.offset<arg::dev_velo_states>(), host_number_of_reconstructed_velo_tracks[0] * sizeof(Velo::State), cudaMemcpyDeviceToHost, stream));

//     // VeloUT tracks
//     cudaCheck(cudaMemcpyAsync(host_atomics_veloUT, arguments.offset<arg::dev_atomics_veloUT>(), arguments.size<arg::dev_atomics_veloUT>(), cudaMemcpyDeviceToHost, stream));
//     cudaCheck(cudaMemcpyAsync(host_veloUT_tracks, arguments.offset<arg::dev_veloUT_tracks>(), arguments.size<arg::dev_veloUT_tracks>(), cudaMemcpyDeviceToHost, stream));

//     // SciFi preprocessing
//     // Estimate cluster count
//     arguments.set_size<arg::dev_scifi_raw_input>(host_scifi_events_size);
//     arguments.set_size<arg::dev_scifi_raw_input_offsets>(host_scifi_event_offsets_size);
//     arguments.set_size<arg::dev_scifi_hit_count>(2 * number_of_events * SciFi::Constants::n_zones + 1);

//     scheduler.setup_next(arguments, sequence_step++);

//     cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_scifi_raw_input>(), host_scifi_events, host_scifi_events_size, cudaMemcpyHostToDevice, stream));
//     cudaCheck(cudaMemcpyAsync(arguments.offset<arg::dev_scifi_raw_input_offsets>(), host_scifi_event_offsets, host_scifi_event_offsets_size * sizeof(uint), cudaMemcpyHostToDevice, stream));
//     cudaCheck(cudaMemsetAsync(arguments.offset<arg::dev_scifi_hit_count>(), 0, arguments.size<arg::dev_scifi_hit_count>(), stream));
//     cudaEventRecord(cuda_generic_event, stream);
//     cudaEventSynchronize(cuda_generic_event);

//     sequence.set_opts<seq::estimate_cluster_count>(dim3(number_of_events), dim3(240), stream);
//     sequence.set_arguments<seq::estimate_cluster_count>(
//       arguments.offset<arg::dev_scifi_raw_input>(),
//       arguments.offset<arg::dev_scifi_raw_input_offsets>(),
//       arguments.offset<arg::dev_scifi_hit_count>(),
//       dev_scifi_geometry
//     );
//     sequence.invoke<seq::estimate_cluster_count>();

//     // Is this needed?
//     cudaEventRecord(cuda_generic_event, stream);
//     cudaEventSynchronize(cuda_generic_event);

//     // Prefix sum of hit count (becomes hit offset)
//     // 1. Reduce
//     // 2. Single block
//     // 3. Scan

//     // Prefix sum: Reduce
//     const uint total_number_of_zones = number_of_events * SciFi::Constants::n_zones;
//     const size_t prefix_sum_auxiliary_array_4_size = (total_number_of_zones + 511) / 512;
//     arguments.set_size<arg::dev_prefix_sum_auxiliary_array_4>(prefix_sum_auxiliary_array_4_size);
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_opts<seq::prefix_sum_reduce_scifi_hits>(dim3(prefix_sum_auxiliary_array_4_size), dim3(256), stream);
//     sequence.set_arguments<seq::prefix_sum_reduce_scifi_hits>(
//       arguments.offset<arg::dev_scifi_hit_count>(),
//       arguments.offset<arg::dev_prefix_sum_auxiliary_array_4>(),
//       total_number_of_zones
//     );
//     sequence.invoke<seq::prefix_sum_reduce_scifi_hits>();

//     // Prefix sum: Single block
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_arguments<seq::prefix_sum_single_block_scifi_hits>(
//       arguments.offset<arg::dev_scifi_hit_count>() + total_number_of_zones,
//       arguments.offset<arg::dev_prefix_sum_auxiliary_array_4>(),
//       prefix_sum_auxiliary_array_4_size
//     );
//     sequence.invoke<seq::prefix_sum_single_block_scifi_hits>();

//     // Prefix sum: Scan
//     scheduler.setup_next(arguments, sequence_step++);
//     const uint pss_scifi_hits_blocks = prefix_sum_auxiliary_array_4_size==1 ? 1 : (prefix_sum_auxiliary_array_4_size-1);
//     sequence.set_opts<seq::prefix_sum_scan_scifi_hits>(dim3(pss_scifi_hits_blocks), dim3(512), stream);
//     sequence.set_arguments<seq::prefix_sum_scan_scifi_hits>(
//       arguments.offset<arg::dev_scifi_hit_count>(),
//       arguments.offset<arg::dev_prefix_sum_auxiliary_array_4>(),
//       total_number_of_zones
//     );
//     sequence.invoke<seq::prefix_sum_scan_scifi_hits>();

//     // Fetch total number of hits
//     cudaCheck(cudaMemcpyAsync(host_accumulated_number_of_scifi_hits,
//       arguments.offset<arg::dev_scifi_hit_count>() + total_number_of_zones,
//       sizeof(uint), cudaMemcpyDeviceToHost, stream));
//     cudaEventRecord(cuda_generic_event, stream);
//     cudaEventSynchronize(cuda_generic_event);
    
//     // Raw Bank Decoder
//     const uint32_t total_scifi_hits_size = 11 * host_accumulated_number_of_scifi_hits[0];
//     arguments.set_size<arg::dev_scifi_hits>(total_scifi_hits_size);

//     scheduler.setup_next(arguments, sequence_step++);

//     sequence.set_opts<seq::raw_bank_decoder>(dim3(number_of_events), dim3(240), stream);
//     sequence.set_arguments<seq::raw_bank_decoder>(
//       arguments.offset<arg::dev_scifi_raw_input>(),
//       arguments.offset<arg::dev_scifi_raw_input_offsets>(),
//       arguments.offset<arg::dev_scifi_hit_count>(),
//       arguments.offset<arg::dev_scifi_hits>(),
//       dev_scifi_geometry
//     );

//     sequence.invoke<seq::raw_bank_decoder>();

//     // SciFi hit sorting by x
//     arguments.set_size<arg::dev_scifi_hit_permutations>(*host_accumulated_number_of_scifi_hits);
//     scheduler.setup_next(arguments, sequence_step++);
//     sequence.set_opts<seq::scifi_sort_by_x>(dim3(number_of_events), dim3(64), stream);
//     sequence.set_arguments<seq::scifi_sort_by_x>(
//       arguments.offset<arg::dev_scifi_hits>(),
//       arguments.offset<arg::dev_scifi_hit_count>(),
//       arguments.offset<arg::dev_scifi_hit_permutations>()
//     );
//     sequence.invoke<seq::scifi_sort_by_x>();

//     // Synchronize
//     cudaEventRecord(cuda_generic_event, stream);
//     cudaEventSynchronize(cuda_generic_event);

//     cudaProfilerStop();
    
//     /* Run Forward on x86 architecture  */
//     std::vector< trackChecker::Tracks > forward_tracks_events;
//     std::vector<uint> host_scifi_hits (total_scifi_hits_size);
//     std::vector<uint> host_scifi_hit_count (2 * number_of_events * SciFi::Constants::n_zones + 1);
    
//     cudaCheck(cudaMemcpyAsync(host_scifi_hits.data(), arguments.offset<arg::dev_scifi_hits>(), arguments.size<arg::dev_scifi_hits>(), cudaMemcpyDeviceToHost, stream ));
//     cudaCheck(cudaMemcpyAsync(host_scifi_hit_count.data(), arguments.offset<arg::dev_scifi_hit_count>(), arguments.size<arg::dev_scifi_hit_count>(), cudaMemcpyDeviceToHost, stream ));
        
//     int rv = run_forward_on_CPU(
//       forward_tracks_events,
//       host_scifi_hits.data(),
//       host_scifi_hit_count.data(),
//       host_velo_tracks_atomics,
//       host_velo_track_hit_number,
//       (uint*)host_velo_states,
//       host_veloUT_tracks,
//       host_atomics_veloUT,
//       number_of_events );
            

//     ///////////////////////
//     // Monte Carlo Check //
//     ///////////////////////

//     if (do_check && i_stream == 0) {
//       if (repetition == 0) { // only check efficiencies once
//         std::cout << "Checking Velo tracks reconstructed on GPU" << std::endl;

//         const std::vector<trackChecker::Tracks> tracks_events = prepareTracks(
//           host_velo_tracks_atomics,
//           host_velo_track_hit_number,
//           host_velo_track_hits,
//           number_of_events);

//         std::string trackType = "Velo";
//         call_pr_checker(
//           tracks_events,
//           folder_name_MC,
//           start_event_offset,
//           trackType
//         );

//         /* CHECKING VeloUT TRACKS */
//         const std::vector< trackChecker::Tracks > veloUT_tracks = prepareVeloUTTracks(
//           host_veloUT_tracks,
//           host_atomics_veloUT,
//           number_of_events
//         );

//          std::cout << "Checking VeloUT tracks reconstructed on GPU" << std::endl;
//          trackType = "VeloUT";
//          call_pr_checker (
//            veloUT_tracks,
//            folder_name_MC,
//            start_event_offset,
//            trackType); 
        
//         /* CHECKING Scifi TRACKS */
//         std::cout << "Checking Forward tracks reconstructed on CPU" << std::endl;
//         trackType = "Forward";
//         call_pr_checker (
//           forward_tracks_events,
//           folder_name_MC,
//           start_event_offset,
//           trackType);
        
//       } // only in first repetition
//     } // do_check
//     // only execute once: not for performance, only for cross-check with GPU results 
//     break;
//   } // repetitions

//   return cudaSuccess;
// }
