#include "Stream.cuh"

cudaError_t Stream::initialize(
  const std::vector<char>& raw_events,
  const std::vector<uint>& event_offsets,
  const std::vector<char>& param_geometry,
  const uint number_of_events,
  const bool param_transmit_host_to_device,
  const bool param_transmit_device_to_host,
  const bool param_do_check,
  const bool param_do_simplified_kalman_filter,
  const bool param_print_individual_rates,
  const std::string param_folder_name_MC,
  const uint param_stream_number
) {
  cudaCheck(cudaStreamCreate(&stream));
  cudaCheck(cudaEventCreate(&cuda_generic_event));
  cudaCheck(cudaEventCreate(&cuda_event_start));
  cudaCheck(cudaEventCreate(&cuda_event_stop));
  stream_number = param_stream_number;
  transmit_host_to_device = param_transmit_host_to_device;
  transmit_device_to_host = param_transmit_device_to_host;
  do_check = param_do_check;
  do_simplified_kalman_filter = param_do_simplified_kalman_filter;
  print_individual_rates = param_print_individual_rates;
  geometry = param_geometry;
  folder_name_MC = param_folder_name_MC;
  
  // Blocks and threads for each algorithm
  const uint prefixSumBlocks = (VeloTracking::n_modules * number_of_events + 511) / 512;
  const uint prefixSumScanBlocks = prefixSumBlocks==1 ? 1 : (prefixSumBlocks-1);

  estimateInputSize.set(     dim3(number_of_events),    dim3(32, 26), stream);
  prefixSumReduce.set(       dim3(prefixSumBlocks),     dim3(256),    stream);
  prefixSumSingleBlock.set(  dim3(1),                   dim3(1024),   stream);
  prefixSumScan.set(         dim3(prefixSumScanBlocks), dim3(512),    stream);
  maskedVeloClustering.set(  dim3(number_of_events),    dim3(256),    stream);
  calculatePhiAndSort.set(   dim3(number_of_events),    dim3(64),     stream);
  searchByTriplet.set(       dim3(number_of_events),    dim3(32),     stream);
  consolidateTracks.set(     dim3(number_of_events),    dim3(32),     stream);
  simplifiedKalmanFilter.set(dim3(number_of_events),    dim3(1024),   stream);

  // Datatypes for definitions below
  // Note: The malloc'ing could be eventually moved to each handler, together
  //       with these datatype definitions.
  //       Keeping it like this is for now is flexible and allows easy testing.
  // 
  // Clustering input
  char* dev_raw_input;
  uint* dev_raw_input_offsets;
  uint* dev_estimated_input_size;
  uint* dev_module_cluster_num;
  uint* dev_module_candidate_num;
  uint* dev_cluster_offset;
  uint* dev_cluster_candidates;
  uint32_t* dev_velo_cluster_container;
  char* dev_velo_geometry;
  // Velo tracking
  TrackHits* dev_tracks;
  uint* dev_tracks_to_follow;
  bool* dev_hit_used;
  int* dev_atomics_storage;
  TrackHits* dev_tracklets;
  uint* dev_weak_tracks;
  Track<do_mc_check>* dev_output_tracks;
  short* dev_h0_candidates;
  short* dev_h2_candidates;
  unsigned short* dev_rel_indices;
  uint* dev_hit_permutation;
  // Velo states
  VeloState* dev_velo_states;

  // velo cluster container contains:
  // - cluster_xs
  // - cluster_ys
  // - cluster_zs
  // - cluster_ids
  // - cluster_phis
  // - temporary
  // 
  // The temporary is required to do the sortinge in an efficient manner
  velo_cluster_container_size = number_of_events * VeloClustering::max_candidates_event * 2 * 6;

  // Data preparation
  // Populate velo geometry
  cudaCheck(cudaMalloc((void**)&dev_velo_geometry, geometry.size()));
  cudaCheck(cudaMemcpyAsync(dev_velo_geometry, geometry.data(), geometry.size(), cudaMemcpyHostToDevice, stream));
  
  // Allocate buffers for algorithms
  // Clustering
  cudaCheck(cudaMalloc((void**)&dev_raw_input, raw_events.size()));
  cudaCheck(cudaMalloc((void**)&dev_raw_input_offsets, event_offsets.size() * sizeof(uint)));
  // DvB: why +2?
  cudaCheck(cudaMalloc((void**)&dev_estimated_input_size, (number_of_events * VeloTracking::n_modules + 2) * sizeof(uint)));
  cudaCheck(cudaMalloc((void**)&dev_cluster_offset, number_of_events * sizeof(uint)));
  cudaCheck(cudaMalloc((void**)&dev_module_cluster_num, number_of_events * VeloTracking::n_modules * sizeof(uint)));
  cudaCheck(cudaMalloc((void**)&dev_module_candidate_num, number_of_events * sizeof(uint)));
  cudaCheck(cudaMalloc((void**)&dev_cluster_candidates, number_of_events * VeloClustering::max_candidates_event * sizeof(uint)));
  cudaCheck(cudaMalloc((void**)&dev_velo_cluster_container, velo_cluster_container_size * sizeof(uint)));

  // phi and sort
  cudaCheck(cudaMalloc((void**)&dev_hit_permutation, VeloTracking::max_number_of_hits_per_event * number_of_events * sizeof(uint)));

  // sbt
  cudaCheck(cudaMalloc((void**)&dev_tracks_to_follow, number_of_events * VeloTracking::ttf_modulo * sizeof(uint)));
  // Note: Don't reuse buffers unless we are on a "performance" branch
  // dev_tracks_to_follow = dev_cluster_candidates;
  
  cudaCheck(cudaMalloc((void**)&dev_tracks, number_of_events * max_tracks_in_event * sizeof(TrackHits)));
  cudaCheck(cudaMalloc((void**)&dev_weak_tracks, VeloTracking::max_number_of_hits_per_event * number_of_events * sizeof(uint)));
  
  cudaCheck(cudaMalloc((void**)&dev_tracklets, VeloTracking::max_number_of_hits_per_event * number_of_events * sizeof(TrackHits)));
  cudaCheck(cudaMalloc((void**)&dev_output_tracks, max_tracks_in_event * number_of_events * sizeof(Track<do_mc_check>)));

  // Note: This is buffer reuse, as the above
  // std::cout << VeloTracking::max_number_of_hits_per_event << " " << number_of_events << " " << sizeof(TrackHits)
  //   << " = " << VeloTracking::max_number_of_hits_per_event * number_of_events * sizeof(TrackHits) << std::endl
  //   << (max_tracks_in_event / 3) << " " << number_of_events << " " << sizeof(Track<do_mc_check>) << " = "
  //   << (max_tracks_in_event / 3) * number_of_events * sizeof(Track<do_mc_check>) << std::endl;

  // const auto tracklets_size = VeloTracking::max_number_of_hits_per_event * number_of_events * sizeof(TrackHits);
  // const auto output_tracks_size = ((tracklets_size / sizeof(Track<do_mc_check>)) + 1) * sizeof(Track<do_mc_check>);
  // std::cout << output_tracks_size << std::endl;

  // cudaCheck(cudaMalloc((void**)&dev_output_tracks, output_tracks_size));
  // dev_tracklets = (TrackHits*) dev_output_tracks;

  cudaCheck(cudaMalloc((void**)&dev_hit_used, VeloTracking::max_number_of_hits_per_event * number_of_events * sizeof(bool)));
  cudaCheck(cudaMalloc((void**)&dev_atomics_storage, number_of_events * atomic_space * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_h0_candidates, 2 * VeloTracking::max_number_of_hits_per_event * number_of_events * sizeof(short)));
  cudaCheck(cudaMalloc((void**)&dev_h2_candidates, 2 * VeloTracking::max_number_of_hits_per_event * number_of_events * sizeof(short)));
  cudaCheck(cudaMalloc((void**)&dev_rel_indices, number_of_events * max_numhits_in_module * sizeof(unsigned short)));

  if (do_simplified_kalman_filter) {
    // simplified kalman filter
    cudaCheck(cudaMalloc((void**)&dev_velo_states, number_of_events * max_tracks_in_event * VeloTracking::states_per_track * sizeof(VeloState)));
  }

  // Memory allocations for host memory (copy back)
  cudaCheck(cudaMallocHost((void**)&host_number_of_tracks_pinned, number_of_events * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_accumulated_tracks, number_of_events * sizeof(int)));
  cudaCheck(cudaMallocHost((void**)&host_tracks_pinned, number_of_events * max_tracks_in_event * sizeof(Track<do_mc_check>)));
  // Pre-populate raw_input data, in case the user requested -a 0
  cudaCheck(cudaMemcpyAsync(dev_raw_input, raw_events.data(), raw_events.size(), cudaMemcpyHostToDevice, stream));
  cudaCheck(cudaMemcpyAsync(dev_raw_input_offsets, event_offsets.data(), event_offsets.size() * sizeof(uint), cudaMemcpyHostToDevice, stream));

  // Prepare kernels
  estimateInputSize.setParameters(
    dev_raw_input,
    dev_raw_input_offsets,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_module_candidate_num,
    dev_cluster_candidates
  );

  prefixSumReduce.setParameters(
    dev_estimated_input_size,
    dev_cluster_offset,
    VeloTracking::n_modules * number_of_events
  );

  prefixSumSingleBlock.setParameters(
    dev_estimated_input_size + VeloTracking::n_modules * number_of_events,
    dev_cluster_offset,
    prefixSumBlocks
  );

  prefixSumScan.setParameters(
    dev_estimated_input_size,
    dev_cluster_offset,
    VeloTracking::n_modules * number_of_events
  );

  maskedVeloClustering.setParameters(
    dev_raw_input,
    dev_raw_input_offsets,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_module_candidate_num,
    dev_cluster_candidates,
    dev_velo_cluster_container,
    dev_velo_geometry
  );

  calculatePhiAndSort.setParameters(
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_velo_cluster_container,
    dev_hit_permutation
  );

  searchByTriplet.setParameters(
    dev_velo_cluster_container,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_tracks,
    dev_tracklets,
    dev_tracks_to_follow,
    dev_weak_tracks,
    dev_hit_used,
    dev_atomics_storage,
    dev_h0_candidates,
    dev_h2_candidates,
    dev_rel_indices
  );
  
  simplifiedKalmanFilter.setParameters(
    dev_velo_cluster_container,
    dev_estimated_input_size,
    dev_atomics_storage,
    dev_tracks,
    dev_velo_states
  );

  consolidateTracks.setParameters(
    dev_atomics_storage,
    dev_tracks,
    dev_output_tracks,
    dev_velo_cluster_container,
    dev_estimated_input_size,
    dev_module_cluster_num
  );

  return cudaSuccess;
}
