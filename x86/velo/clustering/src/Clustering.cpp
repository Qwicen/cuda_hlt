#include "../include/Clustering.h"

std::vector<std::vector<uint32_t>> clustering(
  const std::vector<char>& geometry,
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets,
  const bool assume_never_no_sp
) {
  std::vector<std::vector<uint32_t>> cluster_candidates;
  std::vector<unsigned char> sp_patterns (256, 0);
  std::vector<unsigned char> sp_sizes (256, 0);
  std::vector<float> sp_fx (512, 0);
  std::vector<float> sp_fy (512, 0);

  cache_sp_patterns(sp_patterns, sp_sizes, sp_fx, sp_fy);

  Timer t;

  std::array<unsigned char, VP::NPixelsPerSensor> buffer {};

  // Typecast files and print them
  VeloGeometry g (geometry);
  for (size_t i=1; i<event_offsets.size(); ++i) {
    std::vector<unsigned int> lhcb_ids;

    VeloRawEvent e (events.data() + event_offsets[i-1]);

    auto total_sp_count = 0;

    unsigned int prev_size;
    unsigned int module_sp_count;

    for (unsigned int raw_bank=0; raw_bank<e.number_of_raw_banks; ++raw_bank) {
      std::vector<uint32_t> pixel_idx;

      const auto velo_raw_bank = VeloRawBank(e.payload + e.raw_bank_offset[raw_bank]);
      
      const unsigned int sensor = velo_raw_bank.sensor_index;
      const unsigned int module = sensor / Velo::Constants::n_sensors_per_module;
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
        if (!assume_never_no_sp && no_sp_neighbours) {
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
            const float local_y = (cy + 0.5 + fy) * Velo::Constants::pixel_size;

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
            const float local_y = (cy + 0.5 + fy) * Velo::Constants::pixel_size;

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

          // std::cout << "Cluster (cx, cy): " << cx << ", " << cy << std::endl;

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

          lhcb_ids.emplace_back(get_lhcb_id(cid));
        }
      }
    }

    cluster_candidates.emplace_back(std::move(lhcb_ids));
  }

  t.stop();
  // std::cout << "Classical clustering:" << std::endl
  //   << "Timer: " << t.get() << " s" << std::endl << std::endl;

  return cluster_candidates;
}
