#pragma once

#include "ArgumentManager.cuh"
#include "PrefixSum.cuh"

// TODO: Allow to configure the size of the Prefix Sum sweep at compile time

/**
 * @brief   Defines a Prefix Sum handler.
 * @details A prefix sum is always composed of the same building blocks.
 *          This composited-handler employs reduce, single_block and scan
 *          handlers to provide a prefix sum over the desired datatype.
 *          An auxiliary array must be provided.
 */
#define PREFIX_SUM_ALGORITHM(EXPOSED_TYPE_NAME, DEPENDENCIES)                                        \
  struct EXPOSED_TYPE_NAME {                                                                         \
    constexpr static auto name {#EXPOSED_TYPE_NAME};                                                 \
    constexpr static size_t aux_array_size(size_t array_size) { auto s = (array_size + 511) / 512;   \
                                                                return s > 0 ? s : 1; }              \
    size_t array_size;                                                                               \
    size_t auxiliary_array_size;                                                                     \
    size_t number_of_scan_blocks;                                                                    \
    decltype(make_handler(prefix_sum_reduce)) handler_reduce {prefix_sum_reduce};                    \
    decltype(make_handler(prefix_sum_single_block)) handler_single_block {prefix_sum_single_block};  \
    decltype(make_handler(prefix_sum_scan)) handler_scan {prefix_sum_scan};                          \
    using Arguments = DEPENDENCIES;                                                                  \
    using arguments_t = ArgumentRefManager<Arguments>;                                               \
    void set_size(size_t param_array_size)                                                           \
    {                                                                                                \
      array_size = param_array_size;                                                                 \
      auxiliary_array_size = aux_array_size(array_size);                                             \
      number_of_scan_blocks = auxiliary_array_size == 1 ? 1 : (auxiliary_array_size - 1);            \
    }                                                                                                \
    void set_opts(cudaStream_t& cuda_stream)                                                         \
    {                                                                                                \
      handler_reduce.set_opts(dim3(auxiliary_array_size), dim3(256), cuda_stream);                   \
      handler_single_block.set_opts(dim3(1), dim3(1024), cuda_stream);                               \
      handler_scan.set_opts(dim3(number_of_scan_blocks), dim3(512), cuda_stream);                    \
    }                                                                                                \
    template<typename T>                                                                             \
    void set_arguments(T array, T auxiliary_array)                                                   \
    {                                                                                                \
      handler_reduce.set_arguments(array, auxiliary_array, array_size);                              \
      handler_single_block.set_arguments(array + array_size, auxiliary_array, auxiliary_array_size); \
      handler_scan.set_arguments(array, auxiliary_array, array_size);                                \
    }                                                                                                \
    void invoke()                                                                                    \
    {                                                                                                \
      handler_reduce.invoke();                                                                       \
      handler_single_block.invoke();                                                                 \
      handler_scan.invoke();                                                                         \
    }                                                                                                \
  };

PREFIX_SUM_ALGORITHM(prefix_sum_velo_clusters_t, ARGUMENTS(dev_estimated_input_size, dev_cluster_offset))

PREFIX_SUM_ALGORITHM(
  prefix_sum_velo_track_hit_number_t,
  ARGUMENTS(dev_velo_track_hit_number, dev_prefix_sum_auxiliary_array_2))

PREFIX_SUM_ALGORITHM(
  prefix_sum_ut_track_hit_number_t,
  ARGUMENTS(dev_ut_track_hit_number, dev_prefix_sum_auxiliary_array_5))

PREFIX_SUM_ALGORITHM(prefix_sum_ut_hits_t, ARGUMENTS(dev_ut_hit_offsets, dev_prefix_sum_auxiliary_array_3))

PREFIX_SUM_ALGORITHM(
  prefix_sum_scifi_track_hit_number_t,
  ARGUMENTS(dev_scifi_track_hit_number, dev_prefix_sum_auxiliary_array_6))

PREFIX_SUM_ALGORITHM(prefix_sum_scifi_hits_t, ARGUMENTS(dev_scifi_hit_count, dev_prefix_sum_auxiliary_array_4))
