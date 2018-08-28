#include <inttypes.h>
// unsure why "FTDefinitions.cuh" does not work at the moment.
#include "../../common/include/FTDefinitions.cuh"
__global__ void estimate_cluster_count(uint *ft_event_offsets,  uint *dev_ft_cluster_offsets, uint* dev_ft_cluster_num, char *ft_events);
