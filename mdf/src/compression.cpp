#include <iostream>
#include <cstdio>
#include <assert.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <zlib.h>

#ifdef HAVE_LZMA
#include <lzma.h>
#endif

#ifdef HAVE_LZ4
#include <xxhash.h>
#include <lz4.h>
#include <lz4hc.h>
#endif

#include "compression.hpp"

// The size of the ROOT block framing headers for compression:
// - 3 bytes to identify the compression algorithm and version.
// - 3 bytes to identify the deflated buffer size.
// - 3 bytes to identify the inflated buffer size.
#define HDRSIZE 9

/**
 * Below are the routines for unzipping (inflating) buffers.
 */

namespace {
  static int is_valid_header_zlib(unsigned char* src) { return src[0] == 'Z' && src[1] == 'L' && src[2] == Z_DEFLATED; }

  static int is_valid_header_old(unsigned char* src) { return src[0] == 'C' && src[1] == 'S' && src[2] == Z_DEFLATED; }

  static int is_valid_header_lzma(unsigned char* src) { return src[0] == 'X' && src[1] == 'Z' && src[2] == 0; }

  static int is_valid_header_lz4(unsigned char* src) { return src[0] == 'L' && src[1] == '4'; }

  static int is_valid_header(unsigned char* src)
  {
    return is_valid_header_zlib(src) || is_valid_header_old(src) || is_valid_header_lzma(src) ||
           is_valid_header_lz4(src);
  }

  static const int lzmaHeaderSize = 9;
} // namespace

int Compression::unzip_header(int* srcsize, unsigned char* src, int* tgtsize)
{
  // Reads header envelope, and determines target size.
  // Returns 0 in case of success.

  *srcsize = 0;
  *tgtsize = 0;

  /*   C H E C K   H E A D E R   */
  if (!is_valid_header(src)) {
    fprintf(stderr, "Error unzip_header: error in header.  Values: %x%x\n", src[0], src[1]);
    return 1;
  }

  *srcsize = HDRSIZE + ((long) src[3] | ((long) src[4] << 8) | ((long) src[5] << 16));
  *tgtsize = (long) src[6] | ((long) src[7] << 8) | ((long) src[8] << 16);

  return 0;
}

void Compression::unzip(int* srcsize, unsigned char* src, int* tgtsize, unsigned char* tgt, int* irep)
{
  long isize;
  unsigned char *ibufptr, *obufptr;
  long ibufcnt, obufcnt;

  *irep = 0L;

  /*   C H E C K   H E A D E R   */

  if (*srcsize < HDRSIZE) {
    fprintf(stderr, "unzip: too small source\n");
    return;
  }

  /*   C H E C K   H E A D E R   */
  if (!is_valid_header(src)) {
    fprintf(stderr, "Error unzip: error in header\n");
    return;
  }

  ibufptr = src + HDRSIZE;
  ibufcnt = (long) src[3] | ((long) src[4] << 8) | ((long) src[5] << 16);
  isize = (long) src[6] | ((long) src[7] << 8) | ((long) src[8] << 16);
  obufptr = tgt;
  obufcnt = *tgtsize;

  if (obufcnt < isize) {
    fprintf(stderr, "R__unzip: too small target\n");
    return;
  }

  if (ibufcnt + HDRSIZE != *srcsize) {
    fprintf(stderr, "R__unzip: discrepancy in source length %li %d\n", ibufcnt + HDRSIZE, *srcsize);
    return;
  }

  /* DECOMPRESS DATA */

  /* ZLIB and other standard compression algorithms */
  if (is_valid_header_zlib(src)) {
    unzipZLIB(srcsize, src, tgtsize, tgt, irep);
    return;
#ifdef HAVE_LZMA
  }
  else if (is_valid_header_lzma(src)) {
    unzipLZMA(srcsize, src, tgtsize, tgt, irep);
    return;
#endif
#ifdef HAVE_LZ4
  }
  else if (is_valid_header_lz4(src)) {
    unzipLZ4(srcsize, src, tgtsize, tgt, irep);
    return;
#endif
  }
  else {
    std::cerr << "Unknown compression algorith." << std::endl;
  }
}

void Compression::unzipZLIB(int* srcsize, unsigned char* src, int* tgtsize, unsigned char* tgt, int* irep)
{
  z_stream stream; /* decompression stream */
  int err = 0;

  stream.next_in = (Bytef*) (&src[HDRSIZE]);
  stream.avail_in = (uInt)(*srcsize) - HDRSIZE;
  stream.next_out = (Bytef*) tgt;
  stream.avail_out = (uInt)(*tgtsize);
  stream.zalloc = (alloc_func) 0;
  stream.zfree = (free_func) 0;
  stream.opaque = (voidpf) 0;

  err = inflateInit(&stream);
  if (err != Z_OK) {
    fprintf(stderr, "R__unzip: error %d in inflateInit (zlib)\n", err);
    return;
  }

  while ((err = inflate(&stream, Z_FINISH)) != Z_STREAM_END) {
    if (err != Z_OK) {
      inflateEnd(&stream);
      fprintf(stderr, "R__unzip: error %d in inflate (zlib)\n", err);
      return;
    }
  }

  inflateEnd(&stream);

  *irep = stream.total_out;
  return;
}

#ifdef HAVE_LZMA
void Compression::unzipLZMA(int* srcsize, unsigned char* src, int* tgtsize, unsigned char* tgt, int* irep)
{
  lzma_stream stream = LZMA_STREAM_INIT;
  lzma_ret returnStatus;

  *irep = 0;

  returnStatus = lzma_stream_decoder(&stream, UINT64_MAX, 0U);
  if (returnStatus != LZMA_OK) {
    fprintf(stderr, "R__unzipLZMA: error %d in lzma_stream_decoder\n", returnStatus);
    return;
  }

  stream.next_in = (const uint8_t*) (&src[lzmaHeaderSize]);
  stream.avail_in = (size_t)(*srcsize);
  stream.next_out = (uint8_t*) tgt;
  stream.avail_out = (size_t)(*tgtsize);

  returnStatus = lzma_code(&stream, LZMA_FINISH);
  if (returnStatus != LZMA_STREAM_END) {
    fprintf(stderr, "unzipLZMA: error %d in lzma_code\n", returnStatus);
    lzma_end(&stream);
    return;
  }
  lzma_end(&stream);

  *irep = (int) stream.total_out;
}
#endif

#ifdef HAVE_LZ4
// Header consists of:
// - 2 byte identifier "L4"
// - 1 byte LZ4 version string.
// - 3 bytes of uncompressed size
// - 3 bytes of compressed size
// - 8 byte checksum using xxhash 64.
static const int kChecksumOffset = 2 + 1 + 3 + 3;
static const int kChecksumSize = sizeof(XXH64_canonical_t);
static const int lz4HeaderSize = kChecksumOffset + kChecksumSize;

void Compression::unzipLZ4(int* srcsize, unsigned char* src, int* tgtsize, unsigned char* tgt, int* irep)
{
  // NOTE: We don't check that srcsize / tgtsize is reasonable or within the ROOT-imposed limits.
  // This is assumed to be handled by the upper layers.

  int LZ4_version = LZ4_versionNumber() / (100 * 100);
  *irep = 0;
  if (src[0] != 'L' || src[1] != '4') {
    fprintf(
      stderr,
      "R__unzipLZ4: algorithm run against buffer with incorrect header (got %d%d; expected %d%d).\n",
      src[0],
      src[1],
      'L',
      '4');
    return;
  }
  if (src[2] != LZ4_version) {
    fprintf(
      stderr,
      "R__unzipLZ4: This version of LZ4 is incompatible with the on-disk version (got %d; expected %d).\n",
      src[2],
      LZ4_version);
    return;
  }

  int inputBufferSize = *srcsize - lz4HeaderSize;

  // TODO: The checksum followed by the decompression means we iterate through the buffer twice.
  // We should perform some performance tests to see whether we can interleave the two -- i.e., at
  // what size of chunks does interleaving (avoiding two fetches from RAM) improve enough for the
  // extra function call costs?  NOTE that ROOT limits the buffer size to 16MB.
  XXH64_hash_t checksumResult = XXH64(src + lz4HeaderSize, inputBufferSize, 0);
  XXH64_hash_t checksumFromFile = XXH64_hashFromCanonical(reinterpret_cast<XXH64_canonical_t*>(src + kChecksumOffset));

  if (checksumFromFile != checksumResult) {
    fprintf(
      stderr,
      "R__unzipLZ4: Buffer corruption error!  Calculated checksum %llu; checksum calculated in the file was %llu.\n",
      checksumResult,
      checksumFromFile);
    return;
  }
  int returnStatus = LZ4_decompress_safe((char*) (&src[lz4HeaderSize]), (char*) (tgt), inputBufferSize, *tgtsize);
  if (returnStatus < 0) {
    fprintf(stderr, "R__unzipLZ4: error in decompression around byte %d out of maximum %d.\n", -returnStatus, *tgtsize);
    return;
  }

  *irep = returnStatus;
}
#endif
