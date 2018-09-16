#include <inttypes.h>
// unsure why "FTDefinitions.cuh" does not work at the moment.
#include "../../common/include/FTDefinitions.cuh"
__global__ void raw_bank_decoder(
  uint *ft_event_offsets,
  uint *dev_ft_cluster_offsets,
  char *ft_events,
  char *ft_clusters,
  uint* ft_cluster_nums,
  uint* ft_cluster_num,
  char *ft_geometry);
