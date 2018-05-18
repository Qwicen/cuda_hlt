#include "EstimateInputSize.cuh"

__global__ void estimate_input_size(
  char* dev_raw_input,
  uint* dev_raw_input_offsets,
  uint* dev_estimated_input_size,
  uint* dev_module_cluster_num,
  uint* dev_event_candidate_num,
  uint32_t* dev_cluster_candidates
) {
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const uint raw_bank_number = threadIdx.y;
  const char* raw_input = dev_raw_input + dev_raw_input_offsets[event_number];
  uint* estimated_input_size = dev_estimated_input_size + event_number * N_MODULES;
  uint* module_cluster_num = dev_module_cluster_num + event_number * N_MODULES;
  uint* event_candidate_num = dev_event_candidate_num + event_number;
  uint* number_of_cluster_candidates = dev_estimated_input_size + number_of_events * N_MODULES + 2;
  uint32_t* cluster_candidates = dev_cluster_candidates + event_number * max_candidates_event;

  // Initialize estimated_input_size, module_cluster_num and dev_module_candidate_num to 0
  for (int i=0; i<(N_MODULES + blockDim.x - 1) / blockDim.x; ++i) {
    const auto index = i*blockDim.x + threadIdx.x;
    if (index < N_MODULES) {
      estimated_input_size[index] = 0;
      module_cluster_num[index] = 0;
    }
  }
  *event_candidate_num = 0;

  __syncthreads();

  // Read raw event
  const auto raw_event = VeloRawEvent(raw_input);
  if (raw_bank_number < raw_event.number_of_raw_banks) {
    // Read raw bank
    const auto raw_bank = VeloRawBank(raw_event.payload + raw_event.raw_bank_offset[raw_bank_number]);
    // if ( threadIdx.x == 0 )
    //   printf("event # = %u, # of raw banks = %u, # of sp's in first raw bank = %u, first word = %u \n", event_number, raw_event.number_of_raw_banks, raw_bank.sp_count, raw_bank.sp_word[0]);
    uint* estimated_module_size = estimated_input_size + (raw_bank.sensor_index >> 2);
    // loop over all super pixels within an event using all threads available in one block
    // one block works on one event
    for (int i=0; i<(raw_bank.sp_count + blockDim.x - 1) / blockDim.x; ++i) {
      const auto sp_index = i*blockDim.x + threadIdx.x;
      if (sp_index < raw_bank.sp_count) {
        // Decode sp
        const uint32_t sp_word = raw_bank.sp_word[sp_index];
	// if ( threadIdx.x == 0 && blockIdx.x == 0 )
	//   printf("sp index = %u, word = %u \n", sp_index, sp_word);
        const uint32_t no_sp_neighbours = sp_word & 0x80000000U;
        const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
        const uint8_t sp = sp_word & 0xFFU;

        if (no_sp_neighbours) {
          // The SP does not have any neighbours
          // The problem is as simple as a lookup pattern
          // It can be implemented in two operations

          // Pattern 0:
          // (x  x)
          //  o  o
          // (x  x
          //  x  x)
          //  
          // Note: Pixel order in sp
          // 0x08 | 0x80
          // 0x04 | 0x40
          // 0x02 | 0x20
          // 0x01 | 0x10
          const bool pattern_0 = sp&0x88 && !(sp&0x44) && sp&0x33;

          // Pattern 1:
          // (x  x
          //  x  x)
          //  o  o
          // (x  x)
          const bool pattern_1 = sp&0xCC && !(sp&0x22) && sp&0x11;
          const uint number_of_clusters = 1 + (pattern_0 | pattern_1);
          
          // Add the found clusters
          atomicAdd(estimated_module_size, number_of_clusters);
        } else {
          // Find candidates that follow this condition:
          // For pixel o, all pixels x should *not* be populated
          // x x
          // o x
          //   x

          // Load required neighbouring pixels in order to check the condition
          // x x x
          // o o x
          // o o x
          // o o x
          // o o x
          //   x x
          // 
          // Use an int for storing and calculating
          // Bit order
          // 
          // 4 10 16
          // 3  9 15
          // 2  8 14
          // 1  7 13
          // 0  6 12
          //    5 11
          //    
          // Bit masks
          // 
          // 0x10 0x0400 0x010000
          // 0x08 0x0200   0x8000
          // 0x04 0x0100   0x4000
          // 0x02   0x80   0x2000
          // 0x01   0x40   0x1000
          //        0x20   0x0800
          uint32_t pixels = sp&0x0F | ((sp&0xF0) << 2);

          // Current row and col
          const uint32_t sp_row = sp_addr & 0x3FU;
          const uint32_t sp_col = sp_addr >> 6;

	  // second loop over all super pixels in this event
	  // to find out whether this can become a cluster
	  for (uint k=0; k<raw_bank.sp_count; ++k) {
            const uint32_t other_sp_word = raw_bank.sp_word[k];
            const uint32_t other_no_sp_neighbours = sp_word & 0x80000000U;
            
            if (!other_no_sp_neighbours) {
              const uint32_t other_sp_addr = (other_sp_word & 0x007FFF00U) >> 8;
              const uint32_t other_sp_row = other_sp_addr & 0x3FU;
              const uint32_t other_sp_col = (other_sp_addr >> 6);
              const uint8_t other_sp = other_sp_word & 0xFFU;

              // Populate pixels
              // Note: Pixel order in sp
              // 0x08 | 0x80
              // 0x04 | 0x40
              // 0x02 | 0x20
              // 0x01 | 0x10
              const bool is_top = other_sp_row==(sp_row+1) && other_sp_col==sp_col;
              const bool is_top_right = other_sp_row==(sp_row+1) && other_sp_col==(sp_col+1);
              const bool is_right = other_sp_row==sp_row && other_sp_col==(sp_col+1);
              const bool is_right_bottom = other_sp_row==(sp_row-1) && other_sp_col==(sp_col+1);
              const bool is_bottom = other_sp_row==(sp_row-1) && other_sp_col==sp_col;

              if (is_top || is_top_right || is_right || is_right_bottom || is_bottom) {
                pixels |= is_top*((other_sp&0x01 | ((other_sp&0x10) << 2)) << 4);
                pixels |= is_top_right*((other_sp&0x01) << 16);
                pixels |= is_right*((other_sp&0x0F) << 12);
                pixels |= is_right_bottom*((other_sp&0x08) << 8);
                pixels |= is_bottom*((other_sp&0x80) >> 2);
              }
            } // super pixel not isolated
          } // second loop over super pixels

          // 16 1024 65536
          //  8  512 32768
          //  4  256 16384
          //  2  128  8192
          //  1   64  4096
          //      32  2048
          //                
          // Look up pattern
          // x x
          // o x
          //   x
          // 
          uint found_cluster_candidates = 0;

          // Apply pattern to each pixel
          // Pixels 0 to 3
          for (int k=0; k<4; ++k) {
            // Note: Pattern is 0x71 because of how the int is done:
            // 
            // 0x10 0x0400 0x010000
            // 0x08 0x0200   0x8000
            // 0x04 0x0100   0x4000
            // 0x02   0x80   0x2000
            // 0x01   0x40   0x1000
            //        0x20   0x0800
            //        
            // 0x71 = 0x01 + 0x10 + 0x20 + 0x40
            const bool isolated = (pixels&(0x71 << (k+1))) == 0;
            const bool is_candidate = (pixels&(0x01 << k)) != 0 && isolated;
            found_cluster_candidates += is_candidate;
            if (is_candidate) {
              auto current_cluster_candidate = atomicAdd(event_candidate_num, 1);
              const uint32_t candidate = (sp_index << 11)
                | (raw_bank_number << 3)
                | k;
              cluster_candidates[current_cluster_candidate] = candidate;
            }
          }

          // Pixels 4 to 7
          for (int k=0; k<4; ++k) {
            // For pixels 4 to 7, we need to shift everything by 6
            const bool isolated = (pixels&(0x71 << (k+7))) == 0;
            const bool is_candidate = (pixels&(0x01 << k+6)) != 0 && isolated;
            found_cluster_candidates += is_candidate;
            if (is_candidate) {
              auto current_cluster_candidate = atomicAdd(event_candidate_num, 1);
              const uint32_t candidate = (sp_index << 11)
                | (raw_bank_number << 3)
                | (k+4);
              cluster_candidates[current_cluster_candidate] = candidate;
            }
          }

          // Add the found cluster candidates
          atomicAdd(estimated_module_size, found_cluster_candidates);
        }
      }
    }
  }
}
