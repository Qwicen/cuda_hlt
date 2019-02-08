#include "../include/Clustering.h"
#include <array>

std::vector<std::vector<uint32_t>> cuda_clustering(
  const std::vector<char>& geometry,
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets,
  const int verbosus)
{
  int verbosity = verbosus;

  std::vector<std::vector<uint32_t>> cluster_candidates_to_return;
  std::vector<unsigned char> sp_patterns(256, 0);
  std::vector<unsigned char> sp_sizes(256, 0);
  std::vector<float> sp_fx(512, 0);
  std::vector<float> sp_fy(512, 0);
  cache_sp_patterns(sp_patterns, sp_sizes, sp_fx, sp_fy);

  int print_times = 10;
  int printed = 0;
  constexpr int max_clustering_iterations = 12;

  auto print_array = [](const std::array<uint32_t, 4>& p, const int row = -1, const int col = -1) {
    for (int r = 0; r < 16; ++r) {
      for (int c = 0; c < 8; ++c) {
        if (r == row && c == col) {
          std::cout << "x";
        }
        else {
          const int temp_sp_col = c / 2;
          const bool temp_pixel = (p[temp_sp_col] >> (16 * (c % 2) + (r % 16))) & 0x01;
          std::cout << temp_pixel;
        }
        if (((c + 1) % 2) == 0) std::cout << " ";
      }
      std::cout << std::endl;
      if (((r + 1) % 4) == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
  };

  // Mask for any one pixel array element's next iteration
  const uint32_t mask_bottom = 0xFFFEFFFF;
  const uint32_t mask_top = 0xFFFF7FFF;
  const uint32_t mask_top_left = 0x7FFF7FFF;
  const uint32_t mask_bottom_right = 0xFFFEFFFE;
  auto current_mask = [&mask_bottom, &mask_bottom_right, &mask_top, &mask_top_left](uint32_t p) {
    return ((p & mask_top) << 1) | ((p & mask_bottom) >> 1) | ((p & mask_bottom_right) << 15) |
           ((p & mask_top_left) >> 15) | (p >> 16) | (p >> 17) | (p << 16) | (p << 17);
  };

  // Mask from a pixel array element on the left
  // to be applied on the pixel array element on the right
  const uint32_t mask_ltr_top_right = 0x7FFF0000;
  auto mask_from_left_to_right = [&mask_ltr_top_right](uint32_t p) {
    return ((p & mask_ltr_top_right) >> 15) | (p >> 16) | (p >> 17);
  };

  // Mask from a pixel array element on the right
  // to be applied on the pixel array element on the left
  const uint32_t mask_rtl_bottom_left = 0x0000FFFE;
  auto mask_from_right_to_left = [&mask_rtl_bottom_left](uint32_t p) {
    return ((p & mask_rtl_bottom_left) << 15) | (p << 16) | (p << 17);
  };

  // Create mask for found clusters
  // o o
  // x o
  //   o
  auto cluster_current_mask = [&mask_bottom_right, &mask_top](uint32_t p) {
    return ((p & mask_top) << 1) | ((p & mask_bottom_right) << 15) | (p << 16) | (p << 17);
  };

  // Require the four pixels of the pattern in order to
  // get the candidates
  auto candidates_current_mask = [&mask_bottom](uint32_t p) {
    return ((p & mask_bottom) >> 1) & ((p & mask_top_left) >> 15) & (p >> 16) & (p >> 17);
  };
  auto candidates_current_mask_with_right_clusters = [&mask_bottom, &mask_rtl_bottom_left](uint32_t p, uint32_t rp) {
    return ((p & mask_bottom) >> 1) & (((p & mask_top_left) >> 15) | (rp << 17)) & ((p >> 16) | (rp << 16)) &
           ((p >> 17) | ((rp & mask_rtl_bottom_left) << 15));
  };

  Timer t;

  // Typecast files and print them
  VeloGeometry g(geometry);
  for (size_t i = 1; i < event_offsets.size(); ++i) {
    std::vector<uint32_t> lhcb_ids;
    unsigned int total_sp_count = 0;
    unsigned int no_sp_count = 0;
    unsigned int approximation_number_of_clusters = 0;

    VeloRawEvent e(events.data() + event_offsets[i - 1]);

    for (unsigned int raw_bank = 0; raw_bank < e.number_of_raw_banks; ++raw_bank) {
      std::vector<uint32_t> cluster_candidates;
      const auto velo_raw_bank = VeloRawBank(e.payload + e.raw_bank_offset[raw_bank]);

      const unsigned int sensor = velo_raw_bank.sensor_index;
      const unsigned int module = sensor / Velo::Constants::n_sensors_per_module;
      const float* ltg = g.ltg + 16 * sensor;

      if (verbosity >= logger::debug) {
        std::cout << "Raw bank " << raw_bank << std::endl;
        for (unsigned int j = 0; j < velo_raw_bank.sp_count; ++j) {
          const uint32_t sp_word = velo_raw_bank.sp_word[j];
          const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
          const uint32_t sp_row = sp_addr & 0x3FU;
          const uint32_t sp_col = (sp_addr >> 6);
          const uint32_t no_sp_neighbours = sp_word & 0x80000000U;
          const uint8_t sp = sp_word & 0xFFU;

          if (!no_sp_neighbours) {
            std::cout << "#, row, col, sp: " << j << " " << sp_row << " " << sp_col << " " << ((int) sp) << std::endl;
          }
        }
      }

      for (unsigned int j = 0; j < velo_raw_bank.sp_count; ++j) {
        const uint32_t sp_word = velo_raw_bank.sp_word[j];
        const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
        const uint32_t sp_row = sp_addr & 0x3FU;
        const uint32_t sp_col = (sp_addr >> 6);
        const uint32_t no_sp_neighbours = sp_word & 0x80000000U;
        const uint8_t sp = sp_word & 0xFFU;

        total_sp_count++;

        // There are no neighbours, so compute the number of pixels of this superpixel
        if (no_sp_neighbours) {
          // // Pattern 0:
          // // (x  x)
          // //  o  o
          // // (x  x
          // //  x  x)
          // //
          // // Note: Pixel order in sp
          // // 0x08 | 0x80
          // // 0x04 | 0x40
          // // 0x02 | 0x20
          // // 0x01 | 0x10
          // const bool pattern_0 = sp&0x88 && !(sp&0x44) && sp&0x33;

          // // Pattern 1:
          // // (x  x
          // //  x  x)
          // //  o  o
          // // (x  x)
          // const bool pattern_1 = sp&0xCC && !(sp&0x22) && sp&0x11;
          // const unsigned int number_of_clusters = 1 + pattern_0 + pattern_1;
          // approximation_number_of_clusters += number_of_clusters;

          const int sp_size = sp_sizes[sp];
          const uint32_t idx = sp_patterns[sp];
          const uint32_t chip = sp_col / (VP::ChipColumns / 2);

          if ((sp_size & 0x0F) <= max_cluster_size) {
            approximation_number_of_clusters++;

            // there is always at least one cluster in the super
            // pixel. look up the pattern and add it.
            const uint32_t row = idx & 0x03U;
            const uint32_t col = (idx >> 2) & 1;
            const uint32_t cx = sp_col * 2 + col;
            const uint32_t cy = sp_row * 4 + row;

            unsigned int cid = get_channel_id(sensor, chip, cx % VP::ChipColumns, cy);

            const float fx = sp_fx[sp * 2];
            const float fy = sp_fy[sp * 2];
            const float local_x = g.local_x[cx] + fx * g.x_pitch[cx];
            const float local_y = (cy + 0.5 + fy) * Velo::Constants::pixel_size;

            const float gx = ltg[0] * local_x + ltg[1] * local_y + ltg[9];
            const float gy = ltg[3] * local_x + ltg[4] * local_y + ltg[10];
            const float gz = ltg[6] * local_x + ltg[7] * local_y + ltg[11];

            lhcb_ids.emplace_back(get_lhcb_id(cid));
          }

          // if there is a second cluster for this pattern
          // add it as well.
          if ((idx & 8) && (((sp_size >> 4) & 0x0F) <= max_cluster_size)) {
            approximation_number_of_clusters++;

            const uint32_t row = (idx >> 4) & 3;
            const uint32_t col = (idx >> 6) & 1;
            const uint32_t cx = sp_col * 2 + col;
            const uint32_t cy = sp_row * 4 + row;

            unsigned int cid = get_channel_id(sensor, chip, cx % VP::ChipColumns, cy);

            const float fx = sp_fx[sp * 2 + 1];
            const float fy = sp_fy[sp * 2 + 1];
            const float local_x = g.local_x[cx] + fx * g.x_pitch[cx];
            const float local_y = (cy + 0.5 + fy) * Velo::Constants::pixel_size;

            const float gx = ltg[0] * local_x + ltg[1] * local_y + ltg[9];
            const float gy = ltg[3] * local_x + ltg[4] * local_y + ltg[10];
            const float gz = ltg[6] * local_x + ltg[7] * local_y + ltg[11];

            lhcb_ids.emplace_back(get_lhcb_id(cid));
          }
        }
        else {
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
          uint32_t pixels = (sp & 0x0F) | ((sp & 0xF0) << 2);

          for (unsigned int k = 0; k < velo_raw_bank.sp_count; ++k) {
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
              const bool is_top = other_sp_row == (sp_row + 1) && other_sp_col == sp_col;
              const bool is_top_right = other_sp_row == (sp_row + 1) && other_sp_col == (sp_col + 1);
              const bool is_right = other_sp_row == sp_row && other_sp_col == (sp_col + 1);
              const bool is_right_bottom = other_sp_row == (sp_row - 1) && other_sp_col == (sp_col + 1);
              const bool is_bottom = other_sp_row == (sp_row - 1) && other_sp_col == sp_col;

              if (is_top || is_top_right || is_right || is_right_bottom || is_bottom) {
                pixels |= is_top * (((other_sp & 0x01) | ((other_sp & 0x10) << 2)) << 4);
                pixels |= is_top_right * ((other_sp & 0x01) << 16);
                pixels |= is_right * ((other_sp & 0x0F) << 12);
                pixels |= is_right_bottom * ((other_sp & 0x08) << 8);
                pixels |= is_bottom * ((other_sp & 0x80) >> 2);
              }
            }
          }

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
          // Apply pattern to each pixel
          // Pixels 0 to 3
          for (int k = 0; k < 4; ++k) {
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
            const bool isolated = (pixels & (0x71 << (k + 1))) == 0;
            const bool is_candidate = (pixels & (0x01 << k)) != 0 && isolated;

            const uint32_t row = sp_row * 4 + k;
            const uint32_t col = sp_col * 2;
            const uint32_t idx = row * 770 + col + 771;

            if (is_candidate) {
              cluster_candidates.push_back(idx);
              // lhcb_ids.emplace_back(idx);
            }
          }

          // Pixels 4 to 7
          for (int k = 0; k < 4; ++k) {
            // For pixels 4 to 7, we need to shift everything by 6
            const bool isolated = (pixels & (0x71 << (k + 7))) == 0;
            const bool is_candidate = (pixels & (0x01 << (k + 6))) != 0 && isolated;

            const uint32_t row = sp_row * 4 + k;
            const uint32_t col = sp_col * 2 + 1;
            const uint32_t idx = row * 770 + col + 771;

            if (is_candidate) {
              cluster_candidates.push_back(idx);
              // lhcb_ids.emplace_back(idx);
            }
          }
        }
      }

      // Use cluster candidates to do the clustering
      for (auto pixel : cluster_candidates) {
        // if (pixel == 53514) {
        //   verbosity = 4;
        // } else {
        //   verbosity = 1;
        // }

        // Load the following SPs,
        // where x is the SP containing the pixel, o are other SPs:
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
        std::array<uint32_t, 4> pixel_array {0, 0, 0, 0};

        // Current pixel row and col
        // Note: In all the code below, row and col are int32_t (not unsigned)
        //       This is not a bug
        const uint32_t row = (pixel - 771) / 770;
        const uint32_t col = (pixel - 771) % 770;
        const int32_t sp_row = row / 4;
        const int32_t sp_col = col / 2;

        const int32_t sp_row_lower_limit = sp_row - 2;
        const int32_t sp_row_upper_limit = sp_row + 1;
        const int32_t sp_col_lower_limit = sp_col - 1;
        const int32_t sp_col_upper_limit = sp_col + 2;

        // Load other SPs
        for (unsigned int k = 0; k < velo_raw_bank.sp_count; ++k) {
          const uint32_t other_sp_word = velo_raw_bank.sp_word[k];
          const uint32_t other_no_sp_neighbours = other_sp_word & 0x80000000U;
          if (!other_no_sp_neighbours) {
            const uint32_t other_sp_addr = (other_sp_word & 0x007FFF00U) >> 8;
            const int32_t other_sp_row = other_sp_addr & 0x3FU;
            const int32_t other_sp_col = (other_sp_addr >> 6);
            const uint8_t other_sp = other_sp_word & 0xFFU;

            if (
              other_sp_row >= sp_row_lower_limit && other_sp_row <= sp_row_upper_limit &&
              other_sp_col >= sp_col_lower_limit && other_sp_col <= sp_col_upper_limit) {
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
              pixel_array[relative_col] |= (other_sp & 0X0F) << (4 * relative_row) | (other_sp & 0XF0)
                                                                                       << (12 + 4 * relative_row);
            }
          }
        }

        // Print buffer we will look at
        const int32_t row_lower_limit = sp_row_lower_limit * 4;
        const int32_t row_upper_limit = (sp_row_upper_limit + 1) * 4;
        const int32_t col_lower_limit = sp_col_lower_limit * 2;
        const int32_t col_upper_limit = (sp_col_upper_limit + 1) * 2;

        // Clustering
        if (verbosity >= logger::debug) {
          std::cout << "pixel array" << std::endl;
          print_array(pixel_array);
        }

        // Cluster datatype
        // This will contain our building cluster
        // Start it with row, col element active
        std::array<uint32_t, 4> cluster {0, 0, 0, 0};
        cluster[1] = (0x01 << (row - row_lower_limit)) << (16 * (col % 2));

        if (verbosity >= logger::debug) {
          std::cout << "cluster" << std::endl;
          print_array(cluster);
        }

        // Current cluster being considered for generating the mask
        std::array<uint32_t, 4> working_cluster {0, 0, 0, 0};
        working_cluster[1] = cluster[1];

        if (verbosity >= logger::debug) {
          std::cout << "working cluster" << std::endl;
          print_array(working_cluster);
        }

        // Delete pixels in cluster from pixels
        pixel_array[1] &= ~cluster[1];

        if (verbosity >= logger::debug) {
          std::cout << "pixel array" << std::endl;
          print_array(pixel_array);
        }

        // Create pixel mask
        std::array<uint32_t, 4> pixel_mask {0, 0, 0, 0};

        for (int clustering_iterations = 0; clustering_iterations < max_clustering_iterations;
             ++clustering_iterations) {
          // Create mask for working cluster
          pixel_mask[0] = current_mask(working_cluster[0]) | mask_from_right_to_left(working_cluster[1]);
          pixel_mask[1] = current_mask(working_cluster[1]) | mask_from_right_to_left(working_cluster[2]) |
                          mask_from_left_to_right(working_cluster[0]);
          pixel_mask[2] = current_mask(working_cluster[2]) | mask_from_right_to_left(working_cluster[3]) |
                          mask_from_left_to_right(working_cluster[1]);
          pixel_mask[3] = current_mask(working_cluster[3]) | mask_from_left_to_right(working_cluster[2]);

          if (verbosity >= logger::debug) {
            std::cout << "pixel mask" << std::endl;
            print_array(pixel_mask);
          }

          // Calculate new elements
          working_cluster[0] = pixel_array[0] & pixel_mask[0];
          working_cluster[1] = pixel_array[1] & pixel_mask[1];
          working_cluster[2] = pixel_array[2] & pixel_mask[2];
          working_cluster[3] = pixel_array[3] & pixel_mask[3];

          if (verbosity >= logger::debug) {
            std::cout << "working cluster" << std::endl;
            print_array(working_cluster);
          }

          if (
            working_cluster[0] == 0 && working_cluster[1] == 0 && working_cluster[2] == 0 && working_cluster[3] == 0) {
            break;
          }

          // Add new elements to cluster
          cluster[0] |= working_cluster[0];
          cluster[1] |= working_cluster[1];
          cluster[2] |= working_cluster[2];
          cluster[3] |= working_cluster[3];

          if (verbosity >= logger::debug) {
            std::cout << "cluster" << std::endl;
            print_array(cluster);
          }

          // Delete elements from pixel array
          pixel_array[0] &= ~cluster[0];
          pixel_array[1] &= ~cluster[1];
          pixel_array[2] &= ~cluster[2];
          pixel_array[3] &= ~cluster[3];

          if (verbosity >= logger::debug) {
            std::cout << "pixel array" << std::endl;
            print_array(pixel_array);
          }
        }

        // Calculate x and y from our formed cluster
        // number of active clusters
        const int n = __builtin_popcount(cluster[0]) + __builtin_popcount(cluster[1]) + __builtin_popcount(cluster[2]) +
                      __builtin_popcount(cluster[3]);

        // Prune repeated clusters
        // Only check for repeated clusters for clusters with at least 3 elements
        bool do_store = true;
        if (n >= 3) {
          if (verbosity >= logger::debug) {
            std::cout << "cluster" << std::endl;
            print_array(cluster, row, col);
          }

          // Apply mask for found clusters
          // o o
          // x o
          //   o
          pixel_mask[0] = cluster_current_mask(cluster[0]);
          pixel_mask[1] = cluster_current_mask(cluster[1]) | mask_from_left_to_right(cluster[0]);
          pixel_mask[2] = cluster_current_mask(cluster[2]) | mask_from_left_to_right(cluster[1]);
          pixel_mask[3] = cluster_current_mask(cluster[3]) | mask_from_left_to_right(cluster[2]);

          if (verbosity >= logger::debug) {
            std::cout << "pixel mask" << std::endl;
            print_array(pixel_mask);
          }

          // Do "and not" with found clusters
          // This should return patterns like these:
          // x x
          //   x
          //   x
          working_cluster[0] = pixel_mask[0] & (~cluster[0]);
          working_cluster[1] = pixel_mask[1] & (~cluster[1]);
          working_cluster[2] = pixel_mask[2] & (~cluster[2]);
          working_cluster[3] = pixel_mask[3] & (~cluster[3]);

          if (verbosity >= logger::debug) {
            std::cout << "working cluster" << std::endl;
            print_array(working_cluster);
          }

          // Require the four pixels of the pattern in order to
          // get the candidates
          std::array<uint32_t, 4> candidates;
          candidates[0] = candidates_current_mask_with_right_clusters(working_cluster[0], working_cluster[1]);
          candidates[1] = candidates_current_mask_with_right_clusters(working_cluster[1], working_cluster[2]);
          candidates[2] = candidates_current_mask_with_right_clusters(working_cluster[2], working_cluster[3]);
          candidates[3] = candidates_current_mask(working_cluster[3]);

          // candidates = candidates "and" clusters, to get the real candidates
          candidates[0] &= cluster[0];
          candidates[1] &= cluster[1];
          candidates[2] &= cluster[2];
          candidates[3] &= cluster[3];

          if (verbosity >= logger::debug) {
            std::cout << "candidates" << std::endl;
            print_array(candidates);
          }

          // Remove our cluster candidate
          const uint32_t working_candidate = (0x01 << (row - row_lower_limit)) << (16 * (col % 2));
          candidates[1] ^= working_candidate;

          if (verbosity >= logger::debug) {
            std::cout << "candidates (without working candidate)" << std::endl;
            print_array(candidates);
          }

          // Check if there is another candidate with precedence
          if (candidates[0] || candidates[1] || candidates[2] || candidates[3]) {
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
            const bool working_candidate_under_threshold = working_candidate < 4096;

            // Smaller row on candidates[1]
            uint32_t smaller_row_pixel_mask =
              working_candidate_under_threshold * (0xFFF & negative_working_candidate_mask) |
              (!working_candidate_under_threshold) * (0xFFF & (negative_working_candidate_mask >> 16));
            smaller_row_pixel_mask |= smaller_row_pixel_mask << 16;

            // In order to do the current pixel mask, add the eventual bigger column
            // ie: (add the second column)
            // oo
            // xo
            // oo
            // oo
            const uint32_t current_pixel_mask = smaller_row_pixel_mask | working_candidate_under_threshold * 0xFFFF0000;

            // Compute do_store
            do_store = (candidates[2] | candidates[3] | (candidates[0] & smaller_row_pixel_mask) |
                        (candidates[1] & current_pixel_mask)) == 0;
          }
        }

        if (do_store) {
          // Added value of all x
          const int x = __builtin_popcount(cluster[0] & 0x0000FFFF) * col_lower_limit +
                        __builtin_popcount(cluster[0] & 0xFFFF0000) * (col_lower_limit + 1) +
                        __builtin_popcount(cluster[1] & 0x0000FFFF) * (col_lower_limit + 2) +
                        __builtin_popcount(cluster[1] & 0xFFFF0000) * (col_lower_limit + 3) +
                        __builtin_popcount(cluster[2] & 0x0000FFFF) * (col_lower_limit + 4) +
                        __builtin_popcount(cluster[2] & 0xFFFF0000) * (col_lower_limit + 5) +
                        __builtin_popcount(cluster[3] & 0x0000FFFF) * (col_lower_limit + 6) +
                        __builtin_popcount(cluster[3] & 0xFFFF0000) * (col_lower_limit + 7);

          // Transpose momentarily clusters to obtain y in an easier way
          const std::array<uint32_t, 4> transposed_clusters = {
            (cluster[0] & 0x000F000F) | ((cluster[1] & 0x000F000F) << 4) | ((cluster[2] & 0x000F000F) << 8) |
              ((cluster[3] & 0x000F000F) << 12),
            ((cluster[0] & 0x00F000F0) >> 4) | (cluster[1] & 0x00F000F0) | ((cluster[2] & 0x00F000F0) << 4) |
              ((cluster[3] & 0x00F000F0) << 8),
            ((cluster[0] & 0x0F000F00) >> 8) | ((cluster[1] & 0x0F000F00) >> 4) | (cluster[2] & 0x0F000F00) |
              ((cluster[3] & 0x0F000F00) << 4),
            ((cluster[0] & 0xF000F000) >> 12) | ((cluster[1] & 0xF000F000) >> 8) | ((cluster[2] & 0xF000F000) >> 4) |
              (cluster[3] & 0xF000F000)};

          // Added value of all y
          const int y = __builtin_popcount(transposed_clusters[0] & 0x11111111) * row_lower_limit +
                        __builtin_popcount(transposed_clusters[0] & 0x22222222) * (row_lower_limit + 1) +
                        __builtin_popcount(transposed_clusters[0] & 0x44444444) * (row_lower_limit + 2) +
                        __builtin_popcount(transposed_clusters[0] & 0x88888888) * (row_lower_limit + 3) +
                        __builtin_popcount(transposed_clusters[1] & 0x11111111) * (row_lower_limit + 4) +
                        __builtin_popcount(transposed_clusters[1] & 0x22222222) * (row_lower_limit + 5) +
                        __builtin_popcount(transposed_clusters[1] & 0x44444444) * (row_lower_limit + 6) +
                        __builtin_popcount(transposed_clusters[1] & 0x88888888) * (row_lower_limit + 7) +
                        __builtin_popcount(transposed_clusters[2] & 0x11111111) * (row_lower_limit + 8) +
                        __builtin_popcount(transposed_clusters[2] & 0x22222222) * (row_lower_limit + 9) +
                        __builtin_popcount(transposed_clusters[2] & 0x44444444) * (row_lower_limit + 10) +
                        __builtin_popcount(transposed_clusters[2] & 0x88888888) * (row_lower_limit + 11) +
                        __builtin_popcount(transposed_clusters[3] & 0x11111111) * (row_lower_limit + 12) +
                        __builtin_popcount(transposed_clusters[3] & 0x22222222) * (row_lower_limit + 13) +
                        __builtin_popcount(transposed_clusters[3] & 0x44444444) * (row_lower_limit + 14) +
                        __builtin_popcount(transposed_clusters[3] & 0x88888888) * (row_lower_limit + 15);

          const unsigned int cx = x / n;
          const unsigned int cy = y / n;

          const float fx = x / static_cast<float>(n) - cx;
          const float fy = y / static_cast<float>(n) - cy;

          // store target (3D point for tracking)
          const uint32_t chip = cx / VP::ChipColumns;
          // LHCb::VPChannelID cid(sensor, chip, cx % VP::ChipColumns, cy);
          unsigned int cid = get_channel_id(sensor, chip, cx % VP::ChipColumns, cy);

          const float local_x = g.local_x[cx] + fx * g.x_pitch[cx];
          const float local_y = (cy + 0.5 + fy) * Velo::Constants::pixel_size;
          const float gx = ltg[0] * local_x + ltg[1] * local_y + ltg[9];
          const float gy = ltg[3] * local_x + ltg[4] * local_y + ltg[10];
          const float gz = ltg[6] * local_x + ltg[7] * local_y + ltg[11];

          const auto lhcb_id = get_lhcb_id(cid);
          lhcb_ids.emplace_back(lhcb_id);
          // lhcb_ids.emplace_back(pixel);
        }
      }
    }

    cluster_candidates_to_return.emplace_back(std::move(lhcb_ids));
  }

  t.stop();

  std::cout << "Cuda clustering:" << std::endl << "Timer: " << t.get() << " s" << std::endl << std::endl;

  return cluster_candidates_to_return;
}
