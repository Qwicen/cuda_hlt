#include "RawBankDecoder.cuh"
#include <stdio.h>
#include "assert.h"

__device__ uint16_t channelInBank(uint16_t c) {
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



__global__ void raw_bank_decoder(uint *ft_event_offsets,  uint *dev_ft_cluster_offsets, char *ft_events, FTLiteCluster *ft_clusters, uint* ft_cluster_nums) {
  // TODO: Optimize parallelization (as in estimate_cluster_count).
  const uint event_id = blockIdx.x;
  //if first thread...
  *(ft_cluster_nums + event_id) = 0;
  __syncthreads();

  const FTChannelID temporary_hardcoded_readoutmap[] = {262144u, 264192u, 266240u, 268288u, 270336u, 278528u, 280576u, 282624u, 284672u, 286720u, 294912u, 296960u, 299008u, 301056u, 303104u, 311296u, 313344u, 315392u, 317440u, 319488u, 327680u, 329728u, 331776u, 333824u, 335872u, 344064u, 346112u, 348160u, 350208u, 352256u, 360448u, 362496u, 364544u, 366592u, 368640u, 376832u, 378880u, 380928u, 382976u, 385024u, 393216u, 395264u, 397312u, 399360u, 401408u, 409600u, 411648u, 413696u, 415744u, 417792u, 425984u, 428032u, 430080u, 432128u, 434176u, 442368u, 444416u, 446464u, 448512u, 450560u, 458752u, 460800u, 462848u, 464896u, 466944u, 475136u, 477184u, 479232u, 481280u, 483328u, 491520u, 493568u, 495616u, 497664u, 499712u, 507904u, 509952u, 512000u, 514048u, 516096u, 524288u, 526336u, 528384u, 530432u, 532480u, 540672u, 542720u, 544768u, 546816u, 548864u, 557056u, 559104u, 561152u, 563200u, 565248u, 573440u, 575488u, 577536u, 579584u, 581632u, 589824u, 591872u, 593920u, 595968u, 598016u, 606208u, 608256u, 610304u, 612352u, 614400u, 622592u, 624640u, 626688u, 628736u, 630784u, 638976u, 641024u, 643072u, 645120u, 647168u, 655360u, 657408u, 659456u, 661504u, 663552u, 671744u, 673792u, 675840u, 677888u, 679936u, 688128u, 690176u, 692224u, 694272u, 696320u, 704512u, 706560u, 708608u, 710656u, 712704u, 720896u, 722944u, 724992u, 727040u, 729088u, 737280u, 739328u, 741376u, 743424u, 745472u, 753664u, 755712u, 757760u, 759808u, 761856u, 770048u, 772096u, 774144u, 776192u, 778240u, 786432u, 788480u, 791040u, 793600u, 796160u, 802816u, 804864u, 807424u, 809984u, 812544u, 819200u, 821248u, 823808u, 826368u, 828928u, 835584u, 837632u, 840192u, 842752u, 845312u, 851968u, 854016u, 856576u, 859136u, 861696u, 868352u, 870400u, 872960u, 875520u, 878080u, 884736u, 886784u, 889344u, 891904u, 894464u, 901120u, 903168u, 905728u, 908288u, 910848u, 917504u, 919552u, 922112u, 924672u, 927232u, 933888u, 935936u, 938496u, 941056u, 943616u, 950272u, 952320u, 954880u, 957440u, 960000u, 966656u, 968704u, 971264u, 973824u, 976384u, 983040u, 985088u, 987648u, 990208u, 992768u, 999424u, 1001472u, 1004032u, 1006592u, 1009152u, 1015808u, 1017856u, 1020416u, 1022976u, 1025536u, 1032192u, 1034240u, 1036800u, 1039360u, 1041920u};

  const auto event = FTRawEvent(ft_events + ft_event_offsets[event_id]);
  assert(event.version == 5u);

  for(size_t i = 0; i < 2;i++)//event.number_of_raw_banks; i++)
  {
    uint start = (i == 0? 0 : event.raw_bank_offset[i-1]);
    FTRawBank rawbank(event.payload + start, event.payload + start + event.raw_bank_offset[i] - 1);

    auto offset = temporary_hardcoded_readoutmap[rawbank.sourceID];
    uint16_t* it = rawbank.data + 2;
    uint16_t* last = rawbank.last;
    printf("start: %u\n", start);
    if (*(last-1) == 0) --last;//Remove padding at the end
    for( ;  it < last; ++it ){ // loop over the clusters
      uint16_t c = *it;
      //printf("%x \n", c);
      /*FTChannelID ch = offset + channelInBank(c);

      if( !cSize(c) || it+1 == last ) { //No size flag or last cluster
        //printf("station %u, layer %u, quarter %u, module %u, mat %u, sipm %u, channel %u\n", ch.station(), ch.layer(), ch.quarter(), ch.module(), ch.mat(), ch.sipm(), ch.channel());

        //FTChannelID tmp = (uint)channelInBank(c);
        //printf("station %u, layer %u, quarter %u, module %u, mat %u, sipm %u, channel %u\n\n", tmp.station(), tmp.layer(), tmp.quarter(), tmp.module(), tmp.mat(), tmp.sipm(), tmp.channel());
        //make_cluster(channel,fraction(c),4);
      } else {//Flagged or not the last one.
        unsigned c2 = *(it+1);
        if( cSize(c2) && getLinkInBank(c) == getLinkInBank(c2) ) {
          //make_clusters(channel,c,c2);
          ++it;
        } else {
          //make_cluster(channel,fraction(c),4);
        }
      }*/


      //printf("station %u, layer %u, quarter %u, module %u, mat %u, sipm %u, channel %u\n", ch.station(), ch.layer(), ch.quarter(), ch.module(), ch.mat(), ch.sipm(), ch.channel());
    }


    printf("global offset: %x \n", event.payload + start - ft_events);
    printf("sourceID: %u \n", rawbank.sourceID);

  }
}
