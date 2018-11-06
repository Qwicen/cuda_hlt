#ifndef RAW_HELPERS_H
#define RAW_HELPERS_H 1

#include <iostream>

namespace LHCb  {

  // Forward declarations
  unsigned int hash32Checksum(const void* ptr, size_t len);
  unsigned int adler32Checksum(unsigned int old, const char *buf, size_t len);

  /// Generate XOR Checksum
  unsigned int genChecksum(int flag, const void* ptr, size_t len);

  bool decompressBuffer(int algtype, unsigned char* tar, size_t tar_len,
                        unsigned char* src, size_t src_len, size_t& new_len);

}

#endif
