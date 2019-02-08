#include "../include/Clustering.h"

std::vector<uint32_t> cuda_array_clustering(
  const std::vector<char>& geometry,
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets)
{
  std::cout << std::endl << "cuda array clustering:" << std::endl;
  std::vector<unsigned char> sp_patterns(256, 0);
  std::vector<unsigned char> sp_sizes(256, 0);
  std::vector<float> sp_fx(512, 0);
  std::vector<float> sp_fy(512, 0);
  std::vector<uint32_t> cluster_candidates_to_return;

  size_t max_stack_size = 0;
  cache_sp_patterns(sp_patterns, sp_sizes, sp_fx, sp_fy);

  int print_times = 10;
  int printed = 0;

  Timer t;

  // Typecast files and print them
  VeloGeometry g(geometry);
  for (size_t i = 1; i < event_offsets.size(); ++i) {

    unsigned int total_number_of_clusters = 0;
    unsigned int total_sp_count = 0;
    unsigned int no_sp_count = 0;
    unsigned int approximation_number_of_clusters = 0;

    VeloRawEvent e(events.data() + event_offsets[i - 1]);

    for (unsigned int raw_bank = 0; raw_bank < e.number_of_raw_banks; ++raw_bank) {
      std::vector<uint32_t> cluster_candidates;
      std::vector<uint8_t> buffer(770 * 258, 0);
      std::vector<uint32_t> pixel_idx;

      const auto velo_raw_bank = VeloRawBank(e.payload + e.raw_bank_offset[raw_bank]);

      const unsigned int sensor = velo_raw_bank.sensor_index;
      const unsigned int module = sensor / Velo::Constants::n_sensors_per_module;
      const float* ltg = g.ltg + 16 * sensor;

      for (unsigned int j = 0; j < velo_raw_bank.sp_count; ++j) {
        const uint32_t sp_word = velo_raw_bank.sp_word[j];
        const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
        const uint32_t sp_row = sp_addr & 0x3FU;
        const uint32_t sp_col = (sp_addr >> 6);
        const uint32_t no_sp_neighbours = sp_word & 0x80000000U;
        const uint8_t sp = sp_word & 0xFFU;

        if (no_sp_neighbours) {
          const int sp_size = sp_sizes[sp];
          const uint32_t idx = sp_patterns[sp];
          const uint32_t chip = sp_col / (VP::ChipColumns / 2);

          if ((sp_size & 0x0F) <= max_cluster_size) {
            approximation_number_of_clusters += 1;
          }

          // if there is a second cluster for this pattern
          // add it as well.
          if ((idx & 8) && (((sp_size >> 4) & 0x0F) <= max_cluster_size)) {
            approximation_number_of_clusters += 1;
          }

          continue; // move on to next super pixel
        }

        // record all pixels.
        for (uint32_t shift = 0; shift < 8; ++shift) {
          const uint8_t pixel = (sp >> shift) & 0x01;
          if (pixel) {
            const uint32_t row = sp_row * 4 + shift % 4;
            const uint32_t col = sp_col * 2 + shift / 4;
            // const uint32_t idx = (col << 8) | row;
            const uint32_t idx = row * 770 + col + 771;

            buffer[idx] = 0x1;
            pixel_idx.push_back(idx);
          }
        }
      }

      // the sensor buffer is filled, perform the clustering on
      // clusters that span several super pixels.
      const unsigned int nidx = pixel_idx.size();
      for (unsigned int irc = 0; irc < nidx; ++irc) {
        const uint32_t idx = pixel_idx[irc];

        // Check just four pixels to know whether this is isolated
        // x x
        // o x
        //   x
        const bool non_isolated = buffer[idx + 770] | buffer[idx + 771] | buffer[idx + 1] | buffer[idx - 769];
        if (!non_isolated) {
          cluster_candidates.push_back(idx);
        }
      }

      // std::cout << "Candidates: ";
      // for (auto idx : cluster_candidates) {
      //   const uint32_t row = (idx - 771) / 770;
      //   const uint32_t col = (idx - 771) % 770;
      //   const uint32_t sp_row = row / 4;
      //   const uint32_t sp_col = col / 2;

      //   std::cout << "(" << sp_row << ", " << sp_col << ") ";
      // }
      // std::cout << std::endl;

      // Loop over all cluster candidates and check they are not repeated
      constexpr uint32_t rows_to_check_bottom = 8;
      constexpr uint32_t rows_to_check_top = 4;
      constexpr uint32_t cols_to_check_left = 2;
      constexpr uint32_t cols_to_check_right = 4;

      for (auto idx : cluster_candidates) {
        if (buffer[idx]) {
          // cluster_candidates_to_return.push_back(idx);
          approximation_number_of_clusters += 1;

          std::vector<uint32_t> stack;
          stack.push_back(idx);

          const uint32_t row = (idx - 771) / 770;
          const uint32_t col = (idx - 771) % 770;
          const uint32_t sp_row = row / 4;
          const uint32_t sp_col = col / 2;

          const int32_t row_lower_limit = sp_row * 4 - rows_to_check_bottom;
          const int32_t row_upper_limit = (sp_row + 1) * 4 + rows_to_check_top;
          const int32_t col_lower_limit = sp_col * 2 - cols_to_check_left;
          const int32_t col_upper_limit = (sp_col + 1) * 2 + cols_to_check_right;

          // if (printed++ < print_times) {
          //   // Print buffer we will look at
          //   std::cout << idx << ", "
          //     << row_lower_limit << ", " << row_upper_limit
          //     << ", " << col_lower_limit << ", " << col_upper_limit
          //     << std::endl;

          //   for (int r=row_lower_limit; r<row_upper_limit; ++r) {
          //     for (int c=col_lower_limit; c<col_upper_limit; ++c) {
          //       const uint32_t i = r * 770 + c + 771;
          //       if (i == idx) {
          //         std::cout << "x";
          //       } else if (r<0 || c<0 || r>255 || c>767) {
          //         std::cout << "0";
          //       } else {
          //         std::cout << ((int) buffer[i]);
          //       }
          //       if (((c + 1) % 2) == 0) std::cout << " ";
          //     }
          //     std::cout << std::endl;
          //     if (((r + 1) % 4) == 0) std::cout << std::endl;
          //   }
          //   std::cout << std::endl;
          // }

          while (!stack.empty()) {
            max_stack_size = std::max(max_stack_size, stack.size());

            uint32_t working_id = stack.back();
            stack.pop_back();
            buffer[working_id] = 0;

            const int32_t row = (working_id - 771) / 770;
            const int32_t col = (working_id - 771) % 770;

            // top
            if (row < row_upper_limit - 1) {
              const uint32_t p_row = row + 1;
              const uint32_t p_col = col;
              const uint32_t p_idx = p_row * 770 + p_col + 771;
              if (buffer[p_idx]) {
                stack.push_back(p_idx);
              }
            }

            // top right
            if (row < row_upper_limit - 1 && col < col_upper_limit - 1) {
              const uint32_t p_row = row + 1;
              const uint32_t p_col = col + 1;
              const uint32_t p_idx = p_row * 770 + p_col + 771;
              if (buffer[p_idx]) {
                stack.push_back(p_idx);
              }
            }

            // right
            if (col < col_upper_limit - 1) {
              const uint32_t p_row = row;
              const uint32_t p_col = col + 1;
              const uint32_t p_idx = p_row * 770 + p_col + 771;
              if (buffer[p_idx]) {
                stack.push_back(p_idx);
              }
            }

            // bottom right
            if (row > row_lower_limit && col < col_upper_limit - 1) {
              const uint32_t p_row = row - 1;
              const uint32_t p_col = col + 1;
              const uint32_t p_idx = p_row * 770 + p_col + 771;
              if (buffer[p_idx]) {
                stack.push_back(p_idx);
              }
            }

            // bottom
            if (row > row_lower_limit) {
              const uint32_t p_row = row - 1;
              const uint32_t p_col = col;
              const uint32_t p_idx = p_row * 770 + p_col + 771;
              if (buffer[p_idx]) {
                stack.push_back(p_idx);
              }
            }

            // bottom left
            if (col > col_lower_limit && row > row_lower_limit) {
              const uint32_t p_row = row - 1;
              const uint32_t p_col = col - 1;
              const uint32_t p_idx = p_row * 770 + p_col + 771;
              if (buffer[p_idx]) {
                stack.push_back(p_idx);
              }
            }

            // left
            if (col > col_lower_limit) {
              const uint32_t p_row = row;
              const uint32_t p_col = col - 1;
              const uint32_t p_idx = p_row * 770 + p_col + 771;
              if (buffer[p_idx]) {
                stack.push_back(p_idx);
              }
            }

            // top left
            if (row < row_upper_limit - 1 && col > col_lower_limit) {
              const uint32_t p_row = row + 1;
              const uint32_t p_col = col - 1;
              const uint32_t p_idx = p_row * 770 + p_col + 771;
              if (buffer[p_idx]) {
                stack.push_back(p_idx);
              }
            }
          }
        }
      }
    }

    // std::cout << approximation_number_of_clusters << std::endl;

    // std::cout << "Found " << approximation_number_of_clusters << " clusters for event " << i
    //   << std::endl;
    cluster_candidates_to_return.push_back(approximation_number_of_clusters);
  }

  t.stop();
  std::cout << "Timer: " << t.get() << " s" << std::endl;

  // std::cout << "Max stack size: " << max_stack_size << std::endl;

  return cluster_candidates_to_return;
}
