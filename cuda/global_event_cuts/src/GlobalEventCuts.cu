#include "GlobalEventCuts.cuh"

__global__ void global_event_cuts(
  char* raw_input,
  uint* raw_input_offsets,
  char* ut_raw_input,
  uint* ut_raw_input_offsets,
  char* scifi_raw_input,
  uint* scifi_raw_input_offsets,
  uint* number_of_selected_events,
  uint* event_list) {

  const uint event_number = blockIdx.x; 
  
  // Check SciFi clusters
  const SciFi::SciFiRawEvent scifi_event(scifi_raw_input + scifi_raw_input_offsets[event_number]); 
  uint n_SciFi_clusters = 0;
  for ( uint i = 0; i < scifi_event.number_of_raw_banks; ++i ) {
    // get bank size in bytes, subtract four bytes for header word
    uint bank_size = scifi_event.raw_bank_offset[i + 1] - scifi_event.raw_bank_offset[i] - 4;
    n_SciFi_clusters += bank_size;
  }
  // Bank size is given in bytes. There are 2 bytes per cluster.     
  // 4 bytes are removed for the header.     
  // Note that this overestimates slightly the number of clusters     
  // due to bank padding in 32b. For v5, it further overestimates the     
  // number of clusters due to the merging of clusters.
  n_SciFi_clusters = n_SciFi_clusters/2 - 2;
  
  if ( n_SciFi_clusters >= maxSciFiUTClusters ) 
    return;

  // Check UT clusters
  const uint32_t ut_event_offset = ut_raw_input_offsets[event_number];
  const UTRawEvent ut_event(ut_raw_input + ut_event_offset);
  uint n_UT_clusters = 0;
  for ( uint i = 0; i < ut_event.number_of_raw_banks; ++i ) {
    const UTRawBank ut_bank = ut_event.getUTRawBank(i);
    n_UT_clusters += ut_bank.number_of_hits;
    if ( n_UT_clusters >= maxSciFiUTClusters )
      return;
  }
  
  if ( n_UT_clusters +  n_SciFi_clusters >= maxSciFiUTClusters )
    return;

  // passed cut
  const int selected_event = atomicAdd(number_of_selected_events, 1);
  event_list[selected_event] = event_number;

}
