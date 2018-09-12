#include "RawBankDecoder.cuh"
#include <stdio.h>
#include "assert.h"

using namespace FT;

__device__ uint32_t channelInBank(uint32_t c) {
  return (c >> FTRawBankParams::cellShift);
}

__device__ uint16_t getLinkInBank(uint16_t c){
  return (c >> FTRawBankParams::linkShift);
}

__device__ int cell(uint16_t c) {
  return (c >> FTRawBankParams::cellShift     ) & FTRawBankParams::cellMaximum;
}

__device__ int fraction(uint16_t c) {
  return (c >> FTRawBankParams::fractionShift ) & FTRawBankParams::fractionMaximum;
}

__device__ bool cSize(uint16_t c) {
  return (c >> FTRawBankParams::sizeShift     ) & FTRawBankParams::sizeMaximum;
}


__global__ void raw_bank_decoder(
  uint *ft_event_offsets,
  uint *ft_cluster_offsets,
  char *ft_events,
  FTLiteCluster *ft_clusters,
  uint* ft_cluster_nums,
  char *geometry
) {
  // TODO: Optimize parallelization (as in estimate_cluster_count).
  const uint event_id = blockIdx.x;
  //if first thread...
  *(ft_cluster_nums + event_id) = 0;
  __syncthreads();

  FTGeometry geom(geometry);

  const auto event = FTRawEvent(ft_events + ft_event_offsets[event_id]);
  const uint rawbank_chunk = (event.number_of_raw_banks + blockDim.x - 1) / blockDim.x; // ceiling int division
  assert(event.version == 5u);

  auto make_cluster = [&](uint32_t chan, uint8_t fraction, uint8_t pseudoSize) {
    uint clusterIndex = atomicAdd(ft_cluster_nums + event_id, 1u) + (event_id == 0? 0 : ft_cluster_offsets[event_id-1]);
    ft_clusters[clusterIndex] = {chan, fraction, pseudoSize};
    //printf("making cluster %u: chan %u \n", clusterIndex, ft_clusters[clusterIndex].channelID.channelID);
  };

  //copied straight from FTRawBankDecoder.cpp
  auto make_clusters = [&](uint32_t firstChannel, uint16_t c, uint16_t c2) {
    unsigned int delta = (cell(c2) - cell(c));

    // fragmented clusters, size > 2*max size
    // only edges were saved, add middles now
    if ( delta  > FTRawBankParams::clusterMaxWidth ) {
      //add the first edge cluster, and then the middle clusters
      for(unsigned int  i = FTRawBankParams::clusterMaxWidth; i < delta ; i+= FTRawBankParams::clusterMaxWidth){
        // all middle clusters will have same size as the first cluster,
        // so re-use the fraction
        make_cluster( firstChannel+i, fraction(c), 0 );
      }
      //add the last edge
      make_cluster  ( firstChannel+delta, fraction(c2), 0 );
    } else { //big cluster size upto size 8
      unsigned int widthClus  =  2 * delta - 1 + fraction(c2);
      make_cluster( firstChannel+(widthClus-1)/2 - int((FTRawBankParams::clusterMaxWidth - 1)/2),
                    (widthClus-1)%2, widthClus );
    }//end if adjacent clusters
  };//End lambda make_clusters

  for(uint i = threadIdx.x; i < event.number_of_raw_banks; i+=event.number_of_raw_banks/rawbank_chunk)
  {
    uint start = (i == 0? 0 : event.raw_bank_offset[i-1]);
    FTRawBank rawbank(event.payload + start, event.payload + event.raw_bank_offset[i]);

    uint16_t* it = rawbank.data + 2;
    uint16_t* last = rawbank.last;
    //printf("start: %u\n", start);
    if (*(last-1) == 0) --last;//Remove padding at the end
    for( ;  it < last; ++it ){ // loop over the clusters
      uint16_t c = *it;
      uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + channelInBank(c);
      //printf("byte %x, cib %u, ch %u, station %u, layer %u, quarter %u, module %u, mat %u, sipm %u, channel %u\n", c, channelInBank(c), ch.channelID, ch.station(), ch.layer(), ch.quarter(), ch.module(), ch.mat(), ch.sipm(), ch.channel());

      if( !cSize(c) || it+1 == last ) { //No size flag or last cluster
        make_cluster(ch, fraction(c), 4);
      } else {//Flagged or not the last one.
        unsigned c2 = *(it+1);
        if( cSize(c2) && getLinkInBank(c) == getLinkInBank(c2) ) {
          make_clusters(ch,c,c2);
          ++it;
        } else {
          make_cluster(ch, fraction(c), 4);
        }
      }
    }
    //printf("global offset: %x \n", event.payload + start - ft_events);
    //printf("sourceID: %u \n", rawbank.sourceID);
  }
}
