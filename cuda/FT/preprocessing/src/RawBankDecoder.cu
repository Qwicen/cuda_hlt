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
  char *ft_clusters,
  uint* ft_cluster_nums,
  uint* ft_cluster_num,
  char *geometry
) {};/*
  // TODO: Optimize parallelization (as in estimate_cluster_count).
  const uint event_id = blockIdx.x;
  //if first thread...
  *(ft_cluster_nums + event_id) = 0;
  __syncthreads();

  FTGeometry geom(geometry);

  const auto event = FTRawEvent(ft_events + ft_event_offsets[event_id]);
  // NO version checking. Be careful, as v5 is assumed.
  //assert(event.version == 5u);


  //Merge of PrStoreFTHit and RawBankDecoder.
  auto make_cluster = [&](uint32_t chan, uint8_t fraction, uint8_t pseudoSize) {
    uint clusterIndex = atomicAdd(ft_cluster_nums + event_id, 1u) + (event_id == 0? 0 : ft_cluster_offsets[event_id-1]);
    atomicAdd(ft_cluster_num, 1u);
    ft_clusters[clusterIndex] = {chan, fraction, pseudoSize};
    // const FT::FTChannelID id = {chan};
    // const uint32_t mat = id.uniqueMat();
    // //if(mat > 1024)
    //   printf("event: %u, channelID: %u, uniqueMat: %u \n", event_id, chan, id.uniqueMat());
    // const uint32_t iQuarter = id.uniqueQuarter();
    // const uint32_t info = (iQuarter>>1) | (((iQuarter<<4)^(iQuarter<<5)^128u) & 128u);
    // const float dxdy = geom.dxdy[mat];
    // const float dzdy = geom.dzdy[mat];
    // const float globaldy = geom.globaldy[mat];
    // float uFromChannel = geom.uBegin[mat] + (2 * id.channel() + 1 +clus.fraction) * geom.halfChannelPitch[mat];
    // if( id.die() ) uFromChannel += geom.dieGap[mat];
    // uFromChannel += id.sipm() * geom.sipmPitch[mat];
    // const float endPointX = geom.mirrorPointX[mat] + geom.ddxX[mat] * uFromChannel;
    // const float endPointY = geom.mirrorPointY[mat] + geom.ddxY[mat] * uFromChannel;
    // const float endPointZ = geom.mirrorPointZ[mat] + geom.ddxZ[mat] * uFromChannel;
    // const float x0 = endPointX - dxdy * endPointY;
    // const float z0 = endPointZ - dzdy * endPointY;
    //
    // //TODO resolve this mess..
    // float yMin = id.isBottom()? endPointY + globaldy : endPointY;
    // float yMax = id.isBottom()? endPointY : endPointY + globaldy;
    // //if(id.isBottom()) std::swap(yMin, yMax);
    // assert( clus.pseudoSize < 9 && "Pseudosize of cluster is > 8. Out of range.");
    // float werrX = invClusRes[clus.pseudoSize];
    //
    // const uint32_t hitIndex = i + event_id == 0? 0 : ft_cluster_nums[event_id-1];
    //ft_hits[hitIndex] = {id, x0, z0, dxdy, dzdy, yMin, yMax, werrX, werrX*werrX, info};



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

  for(uint i = threadIdx.x; i < event.number_of_raw_banks; i += blockDim.x)
  {
    FTRawBank rawbank(event.payload + event.raw_bank_offset[i], event.payload + event.raw_bank_offset[i+1]);

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
}*/
