#include "../include/Clustering.h"

uint32_t get_channel_id(
  unsigned int sensor,
  unsigned int chip,
  unsigned int col,
  unsigned int row
) {
  return (sensor << LHCb::VPChannelID::sensorBits) | (chip << LHCb::VPChannelID::chipBits) | (col << LHCb::VPChannelID::colBits) | row;
}

uint32_t get_lhcb_id(uint32_t cid) {
  return (LHCb::VPChannelID::VP << LHCb::detectorTypeBits) + cid;
}

void cache_sp_patterns(
  unsigned char* sp_patterns,
  unsigned char* sp_sizes,
  float* sp_fx,
  float* sp_fy
) {
  // create a cache for all super pixel cluster patterns.
  // this is an unoptimized 8-way flood fill on the 8 pixels
  // in the super pixel.
  // no point in optimizing as this is called once in
  // initialize() and only takes about 20 us.

  // define deltas to 8-connectivity neighbours
  const int dx[] = {-1, 0, 1, -1, 0, 1, -1, 1};
  const int dy[] = {-1, -1, -1, 1, 1, 1, 0, 0};

  // clustering buffer for isolated superpixels.
  unsigned char sp_buffer[8];

  // SP index buffer and its size for single SP clustering
  unsigned char sp_idx[8];
  unsigned char sp_idx_size = 0;

  // stack and stack pointer for single SP clustering
  unsigned char sp_stack[8];
  unsigned char sp_stack_ptr = 0;

  // loop over all possible SP patterns
  for (unsigned int sp = 0; sp < 256; ++sp) {
    sp_idx_size = 0;
    for (unsigned int shift = 0; shift < 8; ++shift) {
      const unsigned char p = sp & (1 << shift);
      sp_buffer[shift] = p;
      if (p) {
        sp_idx[sp_idx_size++] = shift;
      }
    }

    // loop over pixels in this SP and use them as
    // cluster seeds.
    // note that there are at most two clusters
    // in a single super pixel!
    unsigned char clu_idx = 0;
    for (unsigned int ip = 0; ip < sp_idx_size; ++ip) {
      unsigned char idx = sp_idx[ip];

      if (0 == sp_buffer[idx]) {
        continue;
      }  // pixel is used

      sp_stack_ptr = 0;
      sp_stack[sp_stack_ptr++] = idx;
      sp_buffer[idx] = 0;
      unsigned char x = 0;
      unsigned char y = 0;
      unsigned char n = 0;

      while (sp_stack_ptr) {
        idx = sp_stack[--sp_stack_ptr];
        const unsigned char row = idx % 4;
        const unsigned char col = idx / 4;
        x += col;
        y += row;
        ++n;

        for (unsigned int ni = 0; ni < 8; ++ni) {
          const char ncol = col + dx[ni];
          if (ncol < 0 || ncol > 1) continue;
          const char nrow = row + dy[ni];
          if (nrow < 0 || nrow > 3) continue;
          const unsigned char nidx = (ncol << 2) | nrow;
          if (0 == sp_buffer[nidx]) continue;
          sp_stack[sp_stack_ptr++] = nidx;
          sp_buffer[nidx] = 0;
        }
      }

      const uint32_t cx = x / n;
      const uint32_t cy = y / n;
      const float fx = x / static_cast<float>(n) - cx;
      const float fy = y / static_cast<float>(n) - cy;

      sp_sizes[sp] |= n << (4 * clu_idx);

      // store the centroid pixel
      sp_patterns[sp] |= ((cx << 2) | cy) << 4 * clu_idx;

      // set the two cluster flag if this is the second cluster
      sp_patterns[sp] |= clu_idx << 3;

      // set the pixel fractions
      sp_fx[2 * sp + clu_idx] = fx;
      sp_fy[2 * sp + clu_idx] = fy;

      // increment cluster count. note that this can only become 0 or 1!
      ++clu_idx;
    }
  }
}

void clustering(
  const std::vector<char>& geometry,
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets
) {
  std::vector<unsigned char> sp_patterns (256, 0);
  std::vector<unsigned char> sp_sizes (256, 0);
  std::vector<float> sp_fx (512, 0);
  std::vector<float> sp_fy (512, 0);

  cache_sp_patterns(sp_patterns.data(), sp_sizes.data(), sp_fx.data(), sp_fy.data());

  // Typecast files and print them
  VeloGeometry g (geometry);
  for (size_t i=1; i<event_offsets.size(); ++i) {
    std::vector<unsigned int> lhcb_ids;

    VeloRawEvent e (events.data() + event_offsets[i-1], event_offsets[i] - event_offsets[i-1]);

    auto total_sp_count = 0;

    unsigned int prev_size;
    unsigned int module_sp_count;

    for (unsigned int raw_bank=0; raw_bank<e.number_of_raw_banks; ++raw_bank) {
      std::vector<uint32_t> pixel_idx;
      std::vector<unsigned char> buffer (VP::NPixelsPerSensor, 0);
    
      const auto velo_raw_bank = VeloRawBank(e.payload + e.raw_bank_offset[raw_bank]);
      
      const unsigned int sensor = velo_raw_bank.sensor_index;
      const unsigned int module = sensor / g.number_of_sensors_per_module;
      const float* ltg = g.ltg + 16 * sensor;

      total_sp_count += velo_raw_bank.sp_count;
      
      prev_size = lhcb_ids.size();
      if ((raw_bank % 4) == 0) {
        module_sp_count = 0;
      }

      module_sp_count += velo_raw_bank.sp_count;

      for (unsigned int j=0; j<velo_raw_bank.sp_count; ++j) {
        const uint32_t sp_word = *(velo_raw_bank.sp_word + j);

        uint8_t sp = sp_word & 0xFFU;
        
        // protect against zero super pixels.
        if (0 == sp) { continue; };

        const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
        const uint32_t sp_row = sp_addr & 0x3FU;
        const uint32_t sp_col = (sp_addr >> 6);
        const uint32_t no_sp_neighbours = sp_word & 0x80000000U;

        // if a super pixel is isolated the clustering boils
        // down to a simple pattern look up.
        // don't do this if we run in offline mode where we want to record all
        // contributing channels; in that scenario a few more us are negligible
        // compared to the complication of keeping track of all contributing
        // channel IDs.
        if (no_sp_neighbours) {
          const int sp_size = sp_sizes[sp];
          const uint32_t idx = sp_patterns[sp];
          const uint32_t chip = sp_col / (VP::ChipColumns / 2);

          if ((sp_size & 0x0F) <= max_cluster_size) {
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
            const float local_y = (cy + 0.5 + fy) * g.pixel_size;

            const float gx = ltg[0] * local_x + ltg[1] * local_y + ltg[9];
            const float gy = ltg[3] * local_x + ltg[4] * local_y + ltg[10];
            const float gz = ltg[6] * local_x + ltg[7] * local_y + ltg[11];

            lhcb_ids.emplace_back(get_lhcb_id(cid));
          }

          // if there is a second cluster for this pattern
          // add it as well.
          if ((idx & 8) && (((sp_size >> 4) & 0x0F) <= max_cluster_size)) {
            const uint32_t row = (idx >> 4) & 3;
            const uint32_t col = (idx >> 6) & 1;
            const uint32_t cx = sp_col * 2 + col;
            const uint32_t cy = sp_row * 4 + row;

            unsigned int cid = get_channel_id(sensor, chip, cx % VP::ChipColumns, cy);

            const float fx = sp_fx[sp * 2 + 1];
            const float fy = sp_fy[sp * 2 + 1];
            const float local_x = g.local_x[cx] + fx * g.x_pitch[cx];
            const float local_y = (cy + 0.5 + fy) * g.pixel_size;

            const float gx = ltg[0] * local_x + ltg[1] * local_y + ltg[9];
            const float gy = ltg[3] * local_x + ltg[4] * local_y + ltg[10];
            const float gz = ltg[6] * local_x + ltg[7] * local_y + ltg[11];

            lhcb_ids.emplace_back(get_lhcb_id(cid));
          }

          continue;  // move on to next super pixel
        }

        // this one is not isolated or we are targeting clusters; record all
        // pixels.
        for (uint32_t shift = 0; shift < 8; ++shift) {
          const uint8_t pixel = sp & 1;
          if (pixel) {
            const uint32_t row = sp_row * 4 + shift % 4;
            const uint32_t col = sp_col * 2 + shift / 4;
            const uint32_t idx = (col << 8) | row;
            buffer[idx] = pixel;
            pixel_idx.push_back(idx);
          }
          sp = sp >> 1;
          if (0 == sp) break;
        }
      }

      // the sensor buffer is filled, perform the clustering on
      // clusters that span several super pixels.
      const unsigned int nidx = pixel_idx.size();
      for (unsigned int irc = 0; irc < nidx; ++irc) {

        const uint32_t idx = pixel_idx[irc];
        const uint8_t pixel = buffer[idx];

        if (0 == pixel) continue;  // pixel is used in another cluster

        // 8-way row scan optimized seeded flood fill from here.
        std::vector<uint32_t> stack;

        // mark seed as used
        buffer[idx] = 0;

        // initialize sums
        unsigned int x = 0;
        unsigned int y = 0;
        unsigned int n = 0;

        // push seed on stack
        stack.push_back(idx);

        // as long as the stack is not exhausted:
        // - pop the stack and add popped pixel to cluster
        // - scan the row to left and right, adding set pixels
        //   to the cluster and push set pixels above and below
        //   on the stack (and delete both from the pixel buffer).
        while (!stack.empty()) {

          // pop pixel from stack and add it to cluster
          const uint32_t idx = stack.back();
          stack.pop_back();
          const uint32_t row = idx & 0xFFU;
          const uint32_t col = (idx >> 8) & 0x3FFU;
          x += col;
          y += row;
          ++n;

          // check up and down
          uint32_t u_idx = idx + 1;
          if (row < VP::NRows - 1 && buffer[u_idx]) {
            buffer[u_idx] = 0;
            stack.push_back(u_idx);
          }
          uint32_t d_idx = idx - 1;
          if (row > 0 && buffer[d_idx]) {
            buffer[d_idx] = 0;
            stack.push_back(d_idx);
          }

          // scan row to the right
          for (unsigned int c = col + 1; c < VP::NSensorColumns; ++c) {
            const uint32_t nidx = (c << 8) | row;
            // check up and down
            u_idx = nidx + 1;
            if (row < VP::NRows - 1 && buffer[u_idx]) {
              buffer[u_idx] = 0;
              stack.push_back(u_idx);
            }
            d_idx = nidx - 1;
            if (row > 0 && buffer[d_idx]) {
              buffer[d_idx] = 0;
              stack.push_back(d_idx);
            }
            // add set pixel to cluster or stop scanning
            if (buffer[nidx]) {
              buffer[nidx] = 0;
              x += c;
              y += row;
              ++n;
            } else {
              break;
            }
          }

          // scan row to the left
          for (int c = col - 1; c >= 0; --c) {
            const uint32_t nidx = (c << 8) | row;
            // check up and down
            u_idx = nidx + 1;
            if (row < VP::NRows - 1 && buffer[u_idx]) {
              buffer[u_idx] = 0;
              stack.push_back(u_idx);
            }
            d_idx = nidx - 1;
            if (row > 0 && buffer[d_idx]) {
              buffer[d_idx] = 0;
              stack.push_back(d_idx);
            }
            // add set pixel to cluster or stop scanning
            if (buffer[nidx]) {
              buffer[nidx] = 0;
              x += c;
              y += row;
              ++n;
            } else {
              break;
            }
          }
        }  // while the stack is not empty

        // we are done with this cluster, calculate
        // centroid pixel coordinate and fractions.
        if (n <= max_cluster_size) {
          const unsigned int cx = x / n;
          const unsigned int cy = y / n;
          const float fx = x / static_cast<float>(n) - cx;
          const float fy = y / static_cast<float>(n) - cy;

          // store target (3D point for tracking)
          const uint32_t chip = cx / VP::ChipColumns;
          // LHCb::VPChannelID cid(sensor, chip, cx % VP::ChipColumns, cy);
          unsigned int cid = get_channel_id(sensor, chip, cx % VP::ChipColumns, cy);

          const float local_x = g.local_x[cx] + fx * g.x_pitch[cx];
          const float local_y = (cy + 0.5 + fy) * g.pixel_size;
          const float gx = ltg[0] * local_x + ltg[1] * local_y + ltg[9];
          const float gy = ltg[3] * local_x + ltg[4] * local_y + ltg[10];
          const float gz = ltg[6] * local_x + ltg[7] * local_y + ltg[11];

          lhcb_ids.emplace_back(get_lhcb_id(cid));
        }
      }

      // std::cout << velo_raw_bank.sp_count << ", " << (lhcb_ids.size() - prev_size) << std::endl;
      // if (((raw_bank+1) % 4) == 0) {
      //   std::cout << "module sp count, lhcb ids: " << module_sp_count
      //     << ", " << (lhcb_ids.size() - prev_size)
      //     << std::endl;
      // }
    }

    std::cout << "Found " << lhcb_ids.size() << " clusters for event " << i
      << ", sp count: " << total_sp_count << std::endl;
    // std::cout << "Last IDs: ";
    // for (int i=0; i<10; ++i) {
    //   std::cout << lhcb_ids[lhcb_ids.size() - 1 - i] << ", ";
    // }
    // std::cout << std::endl;
  }
}
