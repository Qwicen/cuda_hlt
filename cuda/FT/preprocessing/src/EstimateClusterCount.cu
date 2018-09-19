#include "EstimateClusterCount.cuh"
#include "assert.h"

using namespace FT;

__global__ void estimate_cluster_count(
  char* dev_ft_raw_input,
  uint* dev_ft_raw_input_offsets,
  uint* dev_ft_hit_count, char* dev_ft_geometry
) {
  const uint event_number = blockIdx.x;

  const FTRawEvent event(dev_ft_raw_input + dev_ft_raw_input_offsets[event_number]);
  const FTGeometry geom(dev_ft_geometry);
  FTHitCount hit_count;
  hit_count.typecast_before_prefix_sum(dev_ft_hit_count, event_number);

  // NO version checking. Be careful, as v5 is assumed.
  // assert(event.version == 5u);

  for(uint rawbank = threadIdx.x; rawbank < geom.number_of_tell40s; rawbank += blockDim.x)
  {
    FTChannelID ch = geom.bank_first_channel[rawbank];
    // Apparently the unique* methods are not designed to start at 0, therefore -16
    uint32_t* hits_layer = hit_count.n_hits_layers + ((ch.uniqueQuarter() - 16) >> 1);
    // approx. 2 bytes per cluster in v5 (overestimates a little). See LHCb::FTDAQ::nbFTClusters (FTDAQHelper.cpp)
    uint32_t estimated_cluster_count = (event.raw_bank_offset[rawbank + 1] - event.raw_bank_offset[rawbank]) / 2 - 2u;
    atomicAdd(hits_layer, estimated_cluster_count);
  }
}
