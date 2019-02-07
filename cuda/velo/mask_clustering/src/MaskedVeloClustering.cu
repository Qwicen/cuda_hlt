#include "MaskedVeloClustering.cuh"

// Mask for any one pixel array element's next iteration
__device__ uint32_t current_mask(uint32_t p) {
  return ((p&VeloClustering::mask_top) << 1)
        | ((p&VeloClustering::mask_bottom) >> 1)
        | ((p&VeloClustering::mask_bottom_right) << 15)
        | ((p&VeloClustering::mask_top_left) >> 15)
        | (p >> 16)
        | (p >> 17)
        | (p << 16)
        | (p << 17);
}

// Mask from a pixel array element on the left
// to be applied on the pixel array element on the right
__device__ uint32_t mask_from_left_to_right(uint32_t p) {
  return ((p&VeloClustering::mask_ltr_top_right) >> 15)
    | (p >> 16)
    | (p >> 17);
}

// Mask from a pixel array element on the right
// to be applied on the pixel array element on the left
__device__ uint32_t mask_from_right_to_left(uint32_t p) {
  return ((p&VeloClustering::mask_rtl_bottom_left) << 15)
    | (p << 16)
    | (p << 17);
}

// Create mask for found clusters
// o o
// x o
//   o
__device__ uint32_t cluster_current_mask(uint32_t p) {
  return ((p&VeloClustering::mask_top) << 1)
        | ((p&VeloClustering::mask_bottom_right) << 15)
        | (p << 16)
        | (p << 17);
}

// Require the four pixels of the pattern in order to
// get the candidates
__device__ uint32_t candidates_current_mask(uint32_t p) {
  return ((p&VeloClustering::mask_bottom) >> 1)
      & ((p&VeloClustering::mask_top_left) >> 15)
      & (p >> 16)
      & (p >> 17);
}

__device__ uint32_t candidates_current_mask_with_right_clusters(
  uint32_t p,
  uint32_t rp
) {
  return ((p&VeloClustering::mask_bottom) >> 1)
      & (((p&VeloClustering::mask_top_left) >> 15) | (rp << 17))
      & ((p >> 16) | (rp << 16))
      & ((p >> 17) | ((rp&VeloClustering::mask_rtl_bottom_left) << 15));
}

__global__ void masked_velo_clustering(
  char* dev_raw_input,
  uint* dev_raw_input_offsets,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  uint* dev_event_candidate_num,
  uint* dev_cluster_candidates,
  uint32_t* dev_velo_cluster_container,
  const uint* dev_event_list,
  uint* dev_event_order,
  char* dev_velo_geometry,
  uint8_t* dev_velo_sp_patterns,
  float* dev_velo_sp_fx,
  float* dev_velo_sp_fy
) {
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x; 
  const uint selected_event_number = dev_event_list[event_number];

  const char* raw_input = dev_raw_input + dev_raw_input_offsets[selected_event_number];
  const uint* module_cluster_start = dev_module_cluster_start + event_number * Velo::Constants::n_modules;
  uint* module_cluster_num = dev_module_cluster_num + event_number * Velo::Constants::n_modules;
  uint number_of_candidates = dev_event_candidate_num[event_number];
  uint32_t* cluster_candidates = (uint32_t*) &dev_cluster_candidates[event_number * VeloClustering::max_candidates_event];

  // Local pointers to dev_velo_cluster_container
  const uint estimated_number_of_clusters = dev_module_cluster_start[Velo::Constants::n_modules * number_of_events];
  float* cluster_xs = (float*) &dev_velo_cluster_container[0];
  float* cluster_ys = (float*) &dev_velo_cluster_container[estimated_number_of_clusters];
  float* cluster_zs = (float*) &dev_velo_cluster_container[2 * estimated_number_of_clusters];
  uint32_t* cluster_ids = (uint32_t*) &dev_velo_cluster_container[3 * estimated_number_of_clusters];

  // Load Velo geometry (assume it is the same for all events)
  const VeloGeometry g (dev_velo_geometry);

  // Read raw event
  const auto raw_event = VeloRawEvent(raw_input);

  // process no neighbour sp
  for (int i=0; i<(raw_event.number_of_raw_banks + blockDim.x - 1) / blockDim.x; ++i) {
    const auto raw_bank_number = i*blockDim.x + threadIdx.x;
    if (raw_bank_number < raw_event.number_of_raw_banks) {
      const auto module_number = raw_bank_number >> 2;
      const uint cluster_start = module_cluster_start[module_number];

      // Read raw bank
      const auto raw_bank = VeloRawBank(raw_event.payload + raw_event.raw_bank_offset[raw_bank_number]);
      const float* ltg = g.ltg + 16 * raw_bank.sensor_index;

      for (int sp_index=0; sp_index<raw_bank.sp_count; ++sp_index) {
        // Decode sp
        const uint32_t sp_word = raw_bank.sp_word[sp_index];
        const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
        const uint32_t no_sp_neighbours = sp_word & 0x80000000U;

        // There are no neighbours, so compute the number of pixels of this superpixel
        if (no_sp_neighbours) {
          // Look up pre-generated patterns
          const int32_t sp_row = sp_addr & 0x3FU;
          const int32_t sp_col = (sp_addr >> 6);
          const uint8_t sp = sp_word & 0xFFU;

          const uint32_t idx = dev_velo_sp_patterns[sp];
          const uint32_t chip = sp_col / (VP::ChipColumns / 2);

          {
            // there is always at least one cluster in the super
            // pixel. look up the pattern and add it.
            const uint32_t row = idx & 0x03U;
            const uint32_t col = (idx >> 2) & 1;
            const uint32_t cx = sp_col * 2 + col;
            const uint32_t cy = sp_row * 4 + row;

            const uint cid = get_channel_id(raw_bank.sensor_index, chip, cx % VP::ChipColumns, cy);

            const float fx = dev_velo_sp_fx[sp * 2];
            const float fy = dev_velo_sp_fy[sp * 2];
            const float local_x = g.local_x[cx] + fx * g.x_pitch[cx];
            const float local_y = (cy + 0.5 + fy) * Velo::Constants::pixel_size;

            const uint cluster_num = atomicAdd(module_cluster_num + module_number, 1);

            const float gx = ltg[0] * local_x + ltg[1] * local_y + ltg[9];
            const float gy = ltg[3] * local_x + ltg[4] * local_y + ltg[10];
            const float gz = ltg[6] * local_x + ltg[7] * local_y + ltg[11];

            cluster_xs[cluster_start + cluster_num] = gx;
            cluster_ys[cluster_start + cluster_num] = gy;
            cluster_zs[cluster_start + cluster_num] = gz;
            cluster_ids[cluster_start + cluster_num] = get_lhcb_id(cid);
          }

          // if there is a second cluster for this pattern
          // add it as well.
          if (idx&8) {
            const uint32_t row = (idx >> 4) & 3;
            const uint32_t col = (idx >> 6) & 1;
            const uint32_t cx = sp_col * 2 + col;
            const uint32_t cy = sp_row * 4 + row;

            uint cid = get_channel_id(raw_bank.sensor_index, chip, cx % VP::ChipColumns, cy);

            const float fx = dev_velo_sp_fx[sp * 2 + 1];
            const float fy = dev_velo_sp_fy[sp * 2 + 1];
            const float local_x = g.local_x[cx] + fx * g.x_pitch[cx];
            const float local_y = (cy + 0.5 + fy) * Velo::Constants::pixel_size;

            const uint cluster_num = atomicAdd(module_cluster_num + module_number, 1);

            const float gx = ltg[0] * local_x + ltg[1] * local_y + ltg[9];
            const float gy = ltg[3] * local_x + ltg[4] * local_y + ltg[10];
            const float gz = ltg[6] * local_x + ltg[7] * local_y + ltg[11];

            cluster_xs[cluster_start + cluster_num] = gx;
            cluster_ys[cluster_start + cluster_num] = gy;
            cluster_zs[cluster_start + cluster_num] = gz;
            cluster_ids[cluster_start + cluster_num] = get_lhcb_id(cid);
          }
        }
      }
    }
  }

  __syncthreads();

  // Process rest of clusters
  for (int i=0; i<(number_of_candidates + blockDim.x - 1) / blockDim.x; ++i) {
    const auto candidate_number = i*blockDim.x + threadIdx.x;
    if (candidate_number < number_of_candidates) {
      const uint32_t candidate = cluster_candidates[candidate_number];
      const uint8_t sp_index = candidate >> 11;
      const uint8_t raw_bank_number = (candidate >> 3) & 0xFF;
      const uint32_t module_number = raw_bank_number >> 2;
      const uint8_t candidate_k = candidate & 0x7;

      assert(raw_bank_number < Velo::Constants::n_sensors);

      const auto raw_bank = VeloRawBank(raw_event.payload + raw_event.raw_bank_offset[raw_bank_number]);
      const float* ltg = g.ltg + 16 * raw_bank.sensor_index;
      const uint32_t sp_word = raw_bank.sp_word[sp_index];
      const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
      // Note: In the code below, row and col are int32_t (not unsigned)
      //       This is not a bug
      const int32_t sp_row = sp_addr & 0x3FU;
      const int32_t sp_col = sp_addr >> 6;

      // Find candidates that follow this condition:
      // For pixel x, all pixels o should *not* be populated
      // o o
      // x o
      //   o

      // Load the following SPs,
      // where x is the SP containing the possible candidates, o are other SPs:
      // oooo
      // oxoo
      // oooo
      // oooo
      // 
      // Each column of SPs are in one uint32_t
      // Order is from left to right
      // 
      // 0: o 1: o 2: o 3: o
      //    o    x    o    o
      //    o    o    o    o
      //    o    o    o    o
      // 
      // Order inside an uint32_t is from bottom to top. Eg. 1:
      // 3: o
      // 2: x
      // 1: o
      // 0: o
      uint32_t pixel_array [3] = {0, 0, 0};

      // sp limits to load
      const int32_t sp_row_lower_limit = sp_row - 2;
      const int32_t sp_row_upper_limit = sp_row + 1;
      const int32_t sp_col_lower_limit = sp_col - 1;
      const int32_t sp_col_upper_limit = sp_col + 1;

      // Row limits
      const int32_t row_lower_limit = sp_row_lower_limit * 4;
      const int32_t col_lower_limit = sp_col_lower_limit * 2;

      // Load SPs
      // Note: We will pick up the current one,
      //       no need to add a special case
      for (uint k=0; k<raw_bank.sp_count; ++k) {
        const uint32_t other_sp_word = raw_bank.sp_word[k];
        const uint32_t other_no_sp_neighbours = other_sp_word & 0x80000000U;
        if (!other_no_sp_neighbours) {
          const uint32_t other_sp_addr = (other_sp_word & 0x007FFF00U) >> 8;
          const int32_t other_sp_row = other_sp_addr & 0x3FU;
          const int32_t other_sp_col = (other_sp_addr >> 6);
          const uint8_t other_sp = other_sp_word & 0xFFU;

          if (other_sp_row >= sp_row_lower_limit
            && other_sp_row <= sp_row_upper_limit
            && other_sp_col >= sp_col_lower_limit
            && other_sp_col <= sp_col_upper_limit
          ) {
            const int relative_row = other_sp_row - sp_row_lower_limit;
            const int relative_col = other_sp_col - sp_col_lower_limit;
            
            // Note: Order is:
            // 15 31
            // 14 30
            // 13 29
            // 12 28
            // 11 27
            // 10 26
            //  9 25
            //  8 24
            //  7 23
            //  6 22
            //  5 21
            //  4 20
            //  3 19
            //  2 18
            //  1 17
            //  0 16
            pixel_array[relative_col] |= (other_sp&0X0F) << (4*relative_row)
                                       | (other_sp&0XF0) << (12 + 4*relative_row);
          }
        }
      }

      // Work with candidate k
      const uint32_t row = sp_row * 4 + (candidate_k % 4);
      const uint32_t col = sp_col * 2 + (candidate_k >= 4);

      // Cluster
      // This will contain our building cluster
      // Start it with row, col element active
      uint32_t cluster [3] = {0, (uint32_t) ((0x01 << (row - row_lower_limit)) << (16 * (col % 2))), 0};
      
      // Current cluster being considered for generating the mask
      uint32_t working_cluster [3] = {0, cluster[1], 0};

      // Delete pixels in cluster from pixels
      pixel_array[1] &= ~cluster[1];

      // Perform actual clustering
      for (int clustering_iterations=0; clustering_iterations<VeloClustering::max_clustering_iterations; ++clustering_iterations) {
        // Create mask for working cluster
        uint32_t pixel_mask [3];
        pixel_mask[0] = current_mask(working_cluster[0])
                      | mask_from_right_to_left(working_cluster[1]);
        pixel_mask[1] = current_mask(working_cluster[1])
                      | mask_from_right_to_left(working_cluster[2])
                      | mask_from_left_to_right(working_cluster[0]);
        pixel_mask[2] = current_mask(working_cluster[2])
                      | mask_from_left_to_right(working_cluster[1]);

        // Calculate new elements
        working_cluster[0] = pixel_array[0] & pixel_mask[0];
        working_cluster[1] = pixel_array[1] & pixel_mask[1];
        working_cluster[2] = pixel_array[2] & pixel_mask[2];

        if (working_cluster[0]==0 && working_cluster[1]==0 && working_cluster[2]==0) {
          break;
        }

        // Add new elements to cluster
        cluster[0] |= working_cluster[0];
        cluster[1] |= working_cluster[1];
        cluster[2] |= working_cluster[2];

        // Delete elements from pixel array
        pixel_array[0] &= ~cluster[0];
        pixel_array[1] &= ~cluster[1];
        pixel_array[2] &= ~cluster[2];
      }

      // Early break: If there are any pixels
      // active in SPs to the right, then
      // there must be another pixel eventually
      // fulfilling the condition
      if (cluster[2]) {
        continue;
      }

      // Calculate x and y from our formed cluster
      // number of active clusters
      const int n = __popc(cluster[0])
                  + __popc(cluster[1]);

      // Prune repeated clusters
      // Only check for repeated clusters for clusters with at least 3 elements
      bool do_store = true;
      if (n >= 3) {
        // Apply mask for found clusters
        // o o
        // x o
        //   o
        uint32_t pixel_mask [4];
        pixel_mask[0] = cluster_current_mask(cluster[0]);
        pixel_mask[1] = cluster_current_mask(cluster[1])
                      | mask_from_left_to_right(cluster[0]);
        pixel_mask[2] = mask_from_left_to_right(cluster[1]);

        // Do "and not" with found clusters
        // This should return patterns like these:
        // x x
        //   x
        //   x
        working_cluster[0] = pixel_mask[0] & (~cluster[0]);
        working_cluster[1] = pixel_mask[1] & (~cluster[1]);
        working_cluster[2] = pixel_mask[2];

        // Require the four pixels of the pattern in order to
        // get the candidates
        uint32_t candidates [2];
        candidates[0] = candidates_current_mask_with_right_clusters(working_cluster[0], working_cluster[1]);
        candidates[1] = candidates_current_mask_with_right_clusters(working_cluster[1], working_cluster[2]);

        // candidates = candidates "and" clusters, to get the real candidates
        candidates[0] &= cluster[0];
        candidates[1] &= cluster[1];

        // Remove our cluster candidate
        const uint32_t working_candidate = (0x01 << (row - row_lower_limit)) << (16 * (col % 2));
        candidates[1] ^= working_candidate;

        // Check if there is another candidate with precedence
        if (candidates[0] || candidates[1]) {
          // Precedence:
          // The current candidate should not be considered if there is another candidate
          // with a smaller row, or a bigger column
          // 
          // In order to calculate the last part, we can use the following trick:
          // In two's complement:
          // 32:  00100000
          // -32: 11100000
          // ~(-32): 00011111 (the mask we want)
          const int32_t negative_working_candidate_mask = ~(-working_candidate);
          const bool working_candidate_under_threshold = working_candidate<4096;
          
          // Smaller row on candidates[1]
          uint32_t smaller_row_pixel_mask = working_candidate_under_threshold * (0xFFF&negative_working_candidate_mask)
            | (!working_candidate_under_threshold) * (0xFFF&(negative_working_candidate_mask>>16));
          smaller_row_pixel_mask |= smaller_row_pixel_mask << 16;

          // In order to do the current pixel mask, add the eventual bigger column
          // ie: (add the second column)
          // oo
          // xo
          // oo
          // oo
          const uint32_t current_pixel_mask = smaller_row_pixel_mask
            | working_candidate_under_threshold * 0xFFFF0000;

          // Compute do_store
          do_store = ((candidates[0]&smaller_row_pixel_mask)
                    | (candidates[1]&current_pixel_mask)) == 0;
        }
      }

      if (do_store) {
        // Added value of all x
        const int x = __popc(cluster[0]&0x0000FFFF)*col_lower_limit
                    + __popc(cluster[0]&0xFFFF0000)*(col_lower_limit+1)
                    + __popc(cluster[1]&0x0000FFFF)*(col_lower_limit+2)
                    + __popc(cluster[1]&0xFFFF0000)*(col_lower_limit+3);

        // Transpose momentarily clusters to obtain y in an easier way
        const uint32_t transposed_clusters [4] = {
          ( cluster[0]&0x000F000F)        | ((cluster[1]&0x000F000F) << 4),
          ((cluster[0]&0x00F000F0) >> 4)  | ( cluster[1]&0x00F000F0)      ,
          ((cluster[0]&0x0F000F00) >> 8)  | ((cluster[1]&0x0F000F00) >> 4),
          ((cluster[0]&0xF000F000) >> 12) | ((cluster[1]&0xF000F000) >> 8)
        };

        // Added value of all y
        const int y = __popc(transposed_clusters[0]&0x11111111)*row_lower_limit
                    + __popc(transposed_clusters[0]&0x22222222)*(row_lower_limit+1)
                    + __popc(transposed_clusters[0]&0x44444444)*(row_lower_limit+2)
                    + __popc(transposed_clusters[0]&0x88888888)*(row_lower_limit+3)
                    + __popc(transposed_clusters[1]&0x11111111)*(row_lower_limit+4)
                    + __popc(transposed_clusters[1]&0x22222222)*(row_lower_limit+5)
                    + __popc(transposed_clusters[1]&0x44444444)*(row_lower_limit+6)
                    + __popc(transposed_clusters[1]&0x88888888)*(row_lower_limit+7)
                    + __popc(transposed_clusters[2]&0x11111111)*(row_lower_limit+8)
                    + __popc(transposed_clusters[2]&0x22222222)*(row_lower_limit+9)
                    + __popc(transposed_clusters[2]&0x44444444)*(row_lower_limit+10)
                    + __popc(transposed_clusters[2]&0x88888888)*(row_lower_limit+11)
                    + __popc(transposed_clusters[3]&0x11111111)*(row_lower_limit+12)
                    + __popc(transposed_clusters[3]&0x22222222)*(row_lower_limit+13)
                    + __popc(transposed_clusters[3]&0x44444444)*(row_lower_limit+14)
                    + __popc(transposed_clusters[3]&0x88888888)*(row_lower_limit+15);

        const uint cx = x / n;
        const uint cy = y / n;

        const float fx = x / static_cast<float>(n) - cx;
        const float fy = y / static_cast<float>(n) - cy;

        // store target (3D point for tracking)
        const uint32_t chip = cx / VP::ChipColumns;

        uint cid = get_channel_id(raw_bank.sensor_index, chip, cx % VP::ChipColumns, cy);

        const float local_x = g.local_x[cx] + fx * g.x_pitch[cx];
        const float local_y = (cy + 0.5 + fy) * Velo::Constants::pixel_size;
        
        const uint cluster_num = atomicAdd(module_cluster_num + module_number, 1);

        const float gx = ltg[0] * local_x + ltg[1] * local_y + ltg[9];
        const float gy = ltg[3] * local_x + ltg[4] * local_y + ltg[10];
        const float gz = ltg[6] * local_x + ltg[7] * local_y + ltg[11];
        
        const uint cluster_start = module_cluster_start[module_number];

        const auto lhcb_id = get_lhcb_id(cid);

        assert((cluster_start + cluster_num) < estimated_number_of_clusters);

        cluster_xs[cluster_start + cluster_num] = gx;
        cluster_ys[cluster_start + cluster_num] = gy;
        cluster_zs[cluster_start + cluster_num] = gz;
        cluster_ids[cluster_start + cluster_num] = lhcb_id;
      }
    }
  }
}
