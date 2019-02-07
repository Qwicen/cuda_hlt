#pragma once

#include "SciFiDefinitions.cuh"
#include "SciFiEventModel.cuh"
#include "Handler.cuh"
#include "ArgumentsCommon.cuh"
#include "ArgumentsSciFi.cuh"

__device__ void make_cluster_v4 (
  const int hit_index,
  const SciFi::SciFiGeometry& geom,
  uint32_t chan,
  uint8_t fraction,
  uint8_t pseudoSize,
  uint32_t uniqueMat,
  SciFi::Hits& hits);

__global__ void scifi_raw_bank_decoder_v4(
  char *scifi_events,
  uint *scifi_event_offsets,
  uint *scifi_hit_count,
  uint *scifi_hits,
  const uint* event_list,
  char *scifi_geometry,
  const float* dev_inv_clus_res);

ALGORITHM(scifi_raw_bank_decoder_v4, scifi_raw_bank_decoder_v4_t,
  ARGUMENTS(dev_scifi_raw_input,
    dev_scifi_raw_input_offsets,
    dev_scifi_hit_count,
    dev_scifi_hits,
    dev_event_list))
