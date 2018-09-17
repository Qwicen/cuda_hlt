#include <inttypes.h>
// unsure why "FTDefinitions.cuh" does not work at the moment.
#include "../../common/include/FTDefinitions.cuh"
__global__ void raw_bank_decoder(
  char *ft_events,
  uint *ft_event_offsets,
  uint *ft_hit_count,
  char* ft_hits,
  char *ft_geometry);
