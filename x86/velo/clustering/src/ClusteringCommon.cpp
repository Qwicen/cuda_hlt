#include "../include/ClusteringCommon.h"

void cache_sp_patterns(
  std::vector<unsigned char>& sp_patterns,
  std::vector<unsigned char>& sp_sizes,
  std::vector<float>& sp_fx,
  std::vector<float>& sp_fy)
{
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
      } // pixel is used

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
