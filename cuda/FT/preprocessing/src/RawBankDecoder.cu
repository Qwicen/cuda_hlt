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
  char *ft_events,
  uint *ft_event_offsets,
  uint *ft_hit_count,
  char *ft_hits,
  char *ft_geometry
) {
  //TODO: maybe not hardcoded, or in another place
  const float invClusRes[] = {1/0.05, 1/0.08, 1/0.11, 1/0.14, 1/0.17, 1/0.20, 1/0.23, 1/0.26, 1/0.29};
  const uint32_t number_of_events = gridDim.x;
  const uint32_t event_number = blockIdx.x;

  FTGeometry geom(ft_geometry);
  const auto event = FTRawEvent(ft_events + ft_event_offsets[event_number]);

  FTHits hits;
  hits.typecast_unsorted(ft_hits, ft_hit_count[number_of_events * FT::number_of_zones]);
  FTHitCount hit_count;
  hit_count.typecast_after_prefix_sum(ft_hit_count, event_number, number_of_events);

  __shared__ uint32_t shared_layer_offsets[FT::number_of_zones];

  for (uint i = threadIdx.x; i < FT::number_of_zones; i += blockDim.x) {
    shared_layer_offsets[i] = hit_count.layer_offsets[i];
  }

  for (uint i = threadIdx.x; i < FT::number_of_zones; i += blockDim.x) {
    hit_count.n_hits_layers[i] = 0;
  }

  __syncthreads();

  //Merge of PrStoreFTHit and RawBankDecoder.
  auto make_cluster = [&](uint32_t chan, uint8_t fraction, uint8_t pseudoSize) {
    const FT::FTChannelID id(chan);

    //Offset to save space in geometry structure, see DumpFTGeometry.cpp
    const uint32_t mat = id.uniqueMat() - 512;
    const uint32_t iQuarter = id.uniqueQuarter() - 16;
    const uint32_t planeCode = id.uniqueLayer() - 4;
    const uint32_t info = (iQuarter>>1) | (((iQuarter<<4)^(iQuarter<<5)^128u) & 128u);
    //See Kernel/LHCbID.h. Maybe no hardcoding?
    const uint32_t lhcbid = (10u << 28) + chan;
    const float dxdy = geom.dxdy[mat];
    const float dzdy = geom.dzdy[mat];
    const float globaldy = geom.globaldy[mat];
    float uFromChannel = geom.uBegin[mat] + (2 * id.channel() + 1 + fraction) * geom.halfChannelPitch[mat];
    if( id.die() ) uFromChannel += geom.dieGap[mat];
    uFromChannel += id.sipm() * geom.sipmPitch[mat];
    const float endPointX = geom.mirrorPointX[mat] + geom.ddxX[mat] * uFromChannel;
    const float endPointY = geom.mirrorPointY[mat] + geom.ddxY[mat] * uFromChannel;
    const float endPointZ = geom.mirrorPointZ[mat] + geom.ddxZ[mat] * uFromChannel;
    const float x0 = endPointX - dxdy * endPointY;
    const float z0 = endPointZ - dzdy * endPointY;

    //ORIGINAL: if(id.isBottom()) std::swap(yMin=endPointX, yMax=endPointY + globaldy);
    float yMin = endPointY + id.isBottom() * globaldy;
    float yMax = endPointY + !id.isBottom() * globaldy;

    assert( pseudoSize < 9 && "Pseudosize of cluster is > 8. Out of range.");
    float werrX = invClusRes[pseudoSize];

    //Unsure why -16 is needed. Either something is wrong or the unique* methods are not designed to start at 0.
    const uint32_t uniqueZone = ((id.uniqueQuarter() - 16) >> 1);
    uint32_t* hits_zone = hit_count.n_hits_layers + uniqueZone;
    uint32_t hitIndex = atomicAdd(hits_zone, 1);
    hitIndex += shared_layer_offsets[uniqueZone];

    hits.x0[hitIndex] = x0;
    hits.z0[hitIndex] = z0;
    hits.w[hitIndex] = werrX * werrX;
    hits.dxdy[hitIndex] = dxdy;
    hits.dzdy[hitIndex] = dzdy;
    hits.yMin[hitIndex] = yMin;
    hits.yMax[hitIndex] = yMax;
    hits.werrX[hitIndex] = werrX;
    hits.coord[hitIndex] = 0;
    hits.LHCbID[hitIndex] = lhcbid;
    hits.planeCode[hitIndex] = planeCode;
    hits.hitZone[hitIndex] = uniqueZone % 2;
    hits.info[hitIndex] = info;
    hits.used[hitIndex] = false;
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

  for(uint i = threadIdx.x; i < event.number_of_raw_banks; i += blockDim.x)
  {
    auto rawbank = event.getFTRawBank(i);
    uint16_t* it = rawbank.data + 2;
    uint16_t* last = rawbank.last;
    if (*(last-1) == 0) --last;//Remove padding at the end
    for( ;  it < last; ++it ){ // loop over the clusters
      uint16_t c = *it;
      uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + channelInBank(c);

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
  }
}
