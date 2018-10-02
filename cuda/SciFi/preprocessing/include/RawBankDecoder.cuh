#include "SciFiDefinitions.cuh"
__global__ void raw_bank_decoder(
  char *scifi_events,
  uint *scifi_event_offsets,
  uint *scifi_hit_count,
  uint *scifi_hits,
  char *scifi_geometry);
