#include "GlobalEventCuts.cuh"

__global__ void global_event_cuts(
  char* raw_input,
  uint* raw_input_offsets,
  char* ut_raw_input,
  uint* ut_raw_input_offsets,
  char* scifi_raw_input,
  uint* scifi_raw_input_offsets) {

  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x; 
  
  // Check SciFi clusters first
  const SciFi::SciFiRawEvent event(scifi_raw_input + scifi_raw_input_offsets[event_number]); 
  uint n_SciFi_clusters = 0;
  for ( uint i = 0; i < event.number_of_raw_banks; ++i ) {
    // get bank size in bytes, subtract four bytes for header word
    uint bank_size = event.raw_bank_offset[i + 1] - event.raw_bank_offset[i] - 4;
    n_SciFi_clusters += bank_size;
  }

  n_SciFi_clusters = n_SciFi_clusters/2 - 2;
  

}
