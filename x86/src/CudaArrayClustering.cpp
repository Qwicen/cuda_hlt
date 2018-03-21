#include "../include/Clustering.h"

std::vector<uint32_t> cuda_array_clustering(
  const std::vector<char>& geometry,
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets
) {
  std::cout << std::endl << "cuda array clustering:" << std::endl;
  std::vector<unsigned char> sp_patterns (256, 0);
  std::vector<unsigned char> sp_sizes (256, 0);
  std::vector<float> sp_fx (512, 0);
  std::vector<float> sp_fy (512, 0);
  std::vector<uint32_t> cluster_candidates;

  cache_sp_patterns(sp_patterns.data(), sp_sizes.data(), sp_fx.data(), sp_fy.data());

  // Typecast files and print them
  VeloGeometry g (geometry);
  for (size_t i=1; i<event_offsets.size(); ++i) {

    unsigned int total_number_of_clusters = 0;
    unsigned int total_sp_count = 0;
    unsigned int no_sp_count = 0;
    unsigned int approximation_number_of_clusters = 0;

    VeloRawEvent e (events.data() + event_offsets[i-1], event_offsets[i] - event_offsets[i-1]);

    for (unsigned int raw_bank=0; raw_bank<e.number_of_raw_banks; ++raw_bank) {
      std::vector<uint8_t> buffer (770 * 258, 0);
      std::vector<uint32_t> pixel_idx;

      const auto velo_raw_bank = VeloRawBank(e.payload + e.raw_bank_offset[raw_bank]);
      
      const unsigned int sensor = velo_raw_bank.sensor_index;
      const unsigned int module = sensor / g.number_of_sensors_per_module;
      const float* ltg = g.ltg + 16 * sensor;

      for (unsigned int j=0; j<velo_raw_bank.sp_count; ++j) {
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

          continue;  // move on to next super pixel
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
      const unsigned int prev_number_of_clusters = approximation_number_of_clusters;
      for (unsigned int irc = 0; irc < nidx; ++irc) {
        const uint32_t idx = pixel_idx[irc];

        // Check just four pixels to know whether this is isolated
        // x x
        // o x
        //   x
        const bool non_isolated = buffer[idx+770] | buffer[idx+771] | buffer[idx+1] | buffer[idx-769];
        if (!non_isolated) {
          approximation_number_of_clusters += 1;
          cluster_candidates.push_back(idx);
        }
      }
    }

    std::cout << "Found " << approximation_number_of_clusters << " clusters for event " << i
      << std::endl;
  }

  return cluster_candidates;
}
