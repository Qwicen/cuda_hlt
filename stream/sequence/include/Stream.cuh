#pragma once

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <tuple>

#include "../../../main/include/Common.h"
#include "../../../main/include/CudaCommon.h"
#include "../../../main/include/Logger.h"
#include "../../../main/include/Timer.h"
#include "../../../main/include/Tools.h"
#include "../../../cuda/velo/calculate_phi_and_sort/include/CalculatePhiAndSort.cuh"
#include "../../../cuda/velo/consolidate_tracks/include/ConsolidateTracks.cuh"
#include "../../../cuda/velo/mask_clustering/include/MaskedVeloClustering.cuh"
#include "../../../cuda/velo/mask_clustering/include/EstimateInputSize.cuh"
#include "../../../cuda/velo/prefix_sum/include/PrefixSum.cuh"
#include "../../../cuda/velo/search_by_triplet/include/SearchByTriplet.cuh"
#include "../../../cuda/velo/simplified_kalman_filter/include/VeloKalmanFilter.cuh"
#include "../../handlers/include/Handler.cuh"
#include "../../handlers/include/Argument.cuh"
#include "../../handlers/include/Sequence.cuh"
#include "../../handlers/include/Helper.cuh"
#include "../../memory_manager/include/BaseDynamicScheduler.cuh"

class Timer;

// Unfortunately, enum classes are not bare integers
// and would require an explicit static_cast upon usage.
namespace seq {
enum seq_enum_t {
  estimate_input_size,
  prefix_sum_reduce,
  prefix_sum_single_block,
  prefix_sum_scan,
  masked_velo_clustering,
  calculate_phi_and_sort,
  search_by_triplet,
  copy_and_prefix_sum_single_block,
  copy_velo_track_hit_number,
  prefix_sum_reduce_velo_track_hit_number,
  prefix_sum_single_block_velo_track_hit_number,
  prefix_sum_scan_velo_track_hit_number,
  consolidate_tracks
};
}

namespace arg {
enum arg_enum_t {
  dev_raw_input,
  dev_raw_input_offsets,
  dev_estimated_input_size,
  dev_module_cluster_num,
  dev_module_candidate_num,
  dev_cluster_offset,
  dev_cluster_candidates,
  dev_velo_cluster_container,
  dev_tracks,
  dev_tracks_to_follow,
  dev_hit_used,
  dev_atomics_storage,
  dev_tracklets,
  dev_weak_tracks,
  dev_h0_candidates,
  dev_h2_candidates,
  dev_rel_indices,
  dev_hit_permutation,
  dev_velo_track_hit_number,
  dev_prefix_sum_auxiliary_array_2,
  dev_velo_track_hits,
  dev_velo_states
};
}

struct Stream {
  // Consolidated tracks size is an estimate of an average for each track
  // Note: If we get over that estimate, things can go wrong
  uint consolidate_tracks_average;
  // Stream datatypes
  cudaStream_t stream;
  cudaEvent_t cuda_generic_event;
  cudaEvent_t cuda_event_start;
  cudaEvent_t cuda_event_stop;
  uint stream_number;
  // Launch options
  bool transmit_host_to_device;
  bool transmit_device_to_host;
  bool do_check;
  bool do_simplified_kalman_filter;
  bool print_individual_rates;
  size_t reserve_mb;
  // Varying cluster container size
  uint velo_cluster_container_size;
  // Geometry of Velo detector
  std::vector<char> geometry;

  // Pinned host datatypes
  int* host_number_of_tracks;
  int* host_accumulated_tracks;
  uint* host_velo_track_hit_number;
  Hit<mc_check_enabled>* host_velo_track_hits;
  uint* host_total_number_of_velo_clusters;
  uint* host_number_of_reconstructed_velo_tracks;
  uint* host_accumulated_number_of_hits_in_velo_tracks;
  
  // Sequence
  decltype(generate_sequence(
    generate_handler(estimate_input_size),
    generate_handler(prefix_sum_reduce),
    generate_handler(prefix_sum_single_block),
    generate_handler(prefix_sum_scan),
    generate_handler(masked_velo_clustering),
    generate_handler(calculatePhiAndSort),
    generate_handler(searchByTriplet),
    generate_handler(copy_and_prefix_sum_single_block),
    generate_handler(copy_velo_track_hit_number),
    generate_handler(prefix_sum_reduce),
    generate_handler(prefix_sum_single_block),
    generate_handler(prefix_sum_scan),
    generate_handler(consolidate_tracks)
  )) sequence;

  // Arguments
  decltype(generate_tuple(
    Argument<char>{"dev_raw_input", 0},
    Argument<uint>{"dev_raw_input_offsets", 0},
    Argument<uint>{"dev_estimated_input_size", 0},
    Argument<uint>{"dev_module_cluster_num", 0},
    Argument<uint>{"dev_module_candidate_num", 0},
    Argument<uint>{"dev_cluster_offset", 0},
    Argument<uint>{"dev_cluster_candidates", 0},
    Argument<uint>{"dev_velo_cluster_container", 0},
    Argument<TrackHits>{"dev_tracks", 0},
    Argument<uint>{"dev_tracks_to_follow", 0},
    Argument<bool>{"dev_hit_used", 0},
    Argument<int>{"dev_atomics_storage", 0},
    Argument<TrackHits>{"dev_tracklets", 0},
    Argument<uint>{"dev_weak_tracks", 0},
    Argument<short>{"dev_h0_candidates", 0},
    Argument<short>{"dev_h2_candidates", 0},
    Argument<unsigned short>{"dev_rel_indices", 0},
    Argument<uint>{"dev_hit_permutation", 0},
    Argument<uint>{"dev_velo_track_hit_number", 0},
    Argument<uint>{"dev_prefix_sum_auxiliary_array_2", 0},
    Argument<Hit<mc_check_enabled>>{"dev_velo_track_hits", 0},
    Argument<VeloState>{"dev_velo_states", 0}
  )) arguments;

  // Dynamic scheduler
  BaseDynamicScheduler scheduler;

  // GPU pointers
  char* dev_velo_geometry;
  char* dev_base_pointer;

  // Parameters for certain algorithms
  uint prefixSumBlocks;
  uint prefixSumScanBlocks;

  Stream() = default;

  ~Stream() {}

  std::string folder_name_MC;

  cudaError_t initialize(
    const std::vector<char>& raw_events,
    const std::vector<uint>& event_offsets,
    const std::vector<char>& geometry,
    const uint number_of_events,
    const bool param_transmit_host_to_device,
    const bool param_transmit_device_to_host,
    const bool param_do_check,
    const bool param_do_simplified_kalman_filter,
    const bool param_print_individual_rates,
    const std::string param_folder_name_MC,
    const size_t param_reserve_mb,
    const uint param_stream_number
  );
  
  cudaError_t operator()(
    const char* host_events,
    const uint* host_event_offsets,
    size_t host_events_size,
    size_t host_event_offsets_size,
    uint number_of_events,
    uint number_of_repetitions
  );

  void print_timing(
    const uint number_of_events,
    const std::vector<std::pair<std::string, float>>& times
  );
};
