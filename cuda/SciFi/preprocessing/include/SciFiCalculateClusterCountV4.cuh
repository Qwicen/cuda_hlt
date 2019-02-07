#pragma once

#include "SciFiDefinitions.cuh"
#include "SciFiEventModel.cuh"
#include "Handler.cuh"
#include "ArgumentsCommon.cuh"
#include "ArgumentsSciFi.cuh"

__global__ void scifi_calculate_cluster_count_v4(
  char* scifi_raw_input,
  uint* scifi_raw_input_offsets,
  uint* scifi_hit_count,
  const uint* event_list,
  char* scifi_geometry
);

ALGORITHM(scifi_calculate_cluster_count_v4, scifi_calculate_cluster_count_v4_t,
  ARGUMENTS(dev_scifi_raw_input,
    dev_scifi_raw_input_offsets,
    dev_scifi_hit_count,
    dev_event_list))
