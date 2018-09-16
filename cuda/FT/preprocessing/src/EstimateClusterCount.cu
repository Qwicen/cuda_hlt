#include "EstimateClusterCount.cuh"
#include <stdio.h>
#include "assert.h"

using namespace FT;

__global__ void estimate_cluster_count(char* dev_ft_raw_input, uint* dev_ft_raw_input_offsets, uint* dev_ft_hit_count, char* dev_ft_geometry) {
  // TODO: Optimize parallelization.
  //printf("Preprocessing FT event %u, offset: %u...\n", blockIdx.x, ft_event_offsets[blockIdx.x]);
  const uint event_number = blockIdx.x;

  //if first thread...?
  //*(dev_ft_cluster_offsets + event_id) = 0;
  //*(dev_ft_cluster_num) = 0;
  //__syncthreads();

  const FTRawEvent event(dev_ft_raw_input + dev_ft_raw_input_offsets[event_number]);
  const FTGeometry geom(dev_ft_geometry);
  FTHitCount hit_count;
  hit_count.typecast_before_prefix_sum(dev_ft_hit_count, event_number);
  //const uint rawbank_chunk = (event.number_of_raw_banks + blockDim.x - 1) / blockDim.x; // ceiling int division

  // NO version checking. Be careful, as v5 is assumed.
  //assert(event.version == 5u);

  for(uint rawbank = threadIdx.x; rawbank < event.number_of_raw_banks; rawbank += blockDim.x)
  {
    FTChannelID ch = geom.bank_first_channel[rawbank];
    //Unsure why -16 is needed. Either something is wrong or the unique* methods are not designed to start at 0.
    uint32_t* hits_layer = hit_count.n_hits_layers + ((ch.uniqueQuarter() - 16) >> 1);
    //approx. 2 bytes per cluster in v5 (overestimates a little). See LHCb::FTDAQ::nbFTClusters (FTDAQHelper.cpp)
    uint32_t estimated_cluster_count = (event.raw_bank_offset[rawbank + 1] - event.raw_bank_offset[rawbank]) / 2 - 2u;
    atomicAdd(hits_layer, estimated_cluster_count);
    //atomicAdd(dev_ft_total_hit_count, estimated_cluster_count);
  }
  //printf("Event No: %u, Number of Raw Banks: %u, Version: %u, Total Rawbank Size: %u\n", event_id, event.number_of_raw_banks, event.version, byte_count);
}
