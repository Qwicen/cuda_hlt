#pragma once

#include "SciFiDefinitions.cuh"
#include "SciFiEventModel.cuh"
#include "Handler.cuh"
#include "ArgumentsCommon.cuh"
#include "ArgumentsSciFi.cuh"

__device__ void store_sorted_cluster_reference (
  const SciFi::HitCount& hit_count,
  const uint32_t uniqueMat,
  const uint32_t chan,
  uint32_t* shared_mat_offsets,
  const int raw_bank,
  const int it,
  const int condition_1,
  const int condition_2,
  const int delta,
  SciFi::Hits& hits);

__global__ void scifi_pre_decode(
  char *scifi_events,
  uint *scifi_event_offsets,
  const uint *event_list,
  uint *scifi_hit_count,
  uint *scifi_hits,
  char *scifi_geometry,
  const float* dev_inv_clus_res);

ALGORITHM(scifi_pre_decode, scifi_pre_decode_t,
  ARGUMENTS(dev_scifi_raw_input,
    dev_scifi_raw_input_offsets,
    dev_scifi_hit_count,
    dev_scifi_hits,
    dev_event_list))
