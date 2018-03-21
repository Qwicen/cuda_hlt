#include "../include/Clustering.h"

std::vector<uint32_t> cuda_clustering(
  const std::vector<char>& geometry,
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets
) {
  std::cout << std::endl << "cuda clustering:" << std::endl;
  std::vector<uint32_t> cluster_candidates;

  // Typecast files and print them
  VeloGeometry g (geometry);
  for (size_t i=1; i<event_offsets.size(); ++i) {
    std::vector<unsigned int> lhcb_ids;
    unsigned int total_sp_count = 0;
    unsigned int no_sp_count = 0;
    unsigned int approximation_number_of_clusters = 0;

    VeloRawEvent e (events.data() + event_offsets[i-1], event_offsets[i] - event_offsets[i-1]);

    for (unsigned int raw_bank=0; raw_bank<e.number_of_raw_banks; ++raw_bank) {
      const auto velo_raw_bank = VeloRawBank(e.payload + e.raw_bank_offset[raw_bank]);
      
      const unsigned int sensor = velo_raw_bank.sensor_index;
      const unsigned int module = sensor / g.number_of_sensors_per_module;
      const float* ltg = g.ltg + 16 * sensor;
      
      //std::cout << "Raw bank " << raw_bank << std::endl;
      for (unsigned int j=0; j<velo_raw_bank.sp_count; ++j) {
        const uint32_t sp_word = velo_raw_bank.sp_word[j];
        const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
        const uint32_t sp_row = sp_addr & 0x3FU;
        const uint32_t sp_col = (sp_addr >> 6);
        const uint32_t no_sp_neighbours = sp_word & 0x80000000U;
        const uint8_t sp = sp_word & 0xFFU;

        if (!no_sp_neighbours) {
          //std::cout << "#, row, col, sp: " << j << " " << sp_row << " " << sp_col << " " << ((int) sp) << std::endl;
        }
      }      

      for (unsigned int j=0; j<velo_raw_bank.sp_count; ++j) {
        const uint32_t sp_word = velo_raw_bank.sp_word[j];
        const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
        const uint32_t sp_row = sp_addr & 0x3FU;
        const uint32_t sp_col = (sp_addr >> 6);
        const uint32_t no_sp_neighbours = sp_word & 0x80000000U;
        const uint8_t sp = sp_word & 0xFFU;

        total_sp_count++;

        // There are no neighbours, so compute the number of pixels of this superpixel
        if (no_sp_neighbours) {
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
          const unsigned int number_of_clusters = (sp>0) + pattern_0 + pattern_1;

          approximation_number_of_clusters += number_of_clusters;
        } else {
          // Load neighbouring pixels
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

          for (unsigned int k=0; k<velo_raw_bank.sp_count; ++k) {
            const uint32_t other_sp_word = velo_raw_bank.sp_word[k];
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
            }
          }

          // 16 1024 65536
          //  8  512 32768
          //  4  256 16384
          //  2  128  8192
          //  1   64  4096
          //      32  2048

          // Look up pattern
          // x x
          // o x
          //   x
          // 
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

            const uint32_t row = sp_row * 4 + k;
            const uint32_t col = sp_col * 2;
            const uint32_t idx = row * 770 + col + 771;

            if (is_candidate) {
              approximation_number_of_clusters += 1;
              cluster_candidates.push_back(idx);
            }
          }

          // Pixels 4 to 7
          for (int k=0; k<4; ++k) {
            // For pixels 4 to 7, we need to shift everything by 6
            const bool isolated = (pixels&(0x71 << (k+7))) == 0;
            const bool is_candidate = (pixels&(0x01 << k+6)) != 0 && isolated;

            const uint32_t row = sp_row * 4 + k;
            const uint32_t col = sp_col * 2 + 1;
            const uint32_t idx = row * 770 + col + 771;
            
            if (is_candidate) {
              approximation_number_of_clusters += 1;
              cluster_candidates.push_back(idx);
            }
          }
        }
      }
    }

    std::cout << "Found " << approximation_number_of_clusters << " clusters for event " << i
      << ", sp count: " << total_sp_count
      << ", no sp neighbour %: " << (100.0 * no_sp_count) / ((float) total_sp_count)
      << std::endl;

    // std::cout << "Last IDs: ";
    // for (int i=0; i<10; ++i) {
    //   std::cout << lhcb_ids[lhcb_ids.size() - 1 - i] << ", ";
    // }
    // std::cout << std::endl;
  }

  return cluster_candidates;
}
