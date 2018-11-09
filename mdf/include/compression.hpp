// @(#)root/zip:$Id$
// Author: Sergey Linev   7 July 2014

/*************************************************************************
 * Copyright (C) 1995-2014, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// #include "Compression.h"

/**
 * These are definitions of various free functions for the C-style compression routines in ROOT.
 */

#ifndef COMPRESSION_H
#define COMPRESSION_H 1

namespace Compression {

unsigned long crc32(unsigned long crc, const unsigned char* buf, unsigned int len);

void unzip(int *srcsize, unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep);

int unzip_header(int *srcsize, unsigned char *src, int *tgtsize);

void unzipZLIB(int *srcsize, unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep);

#ifdef HAVE_LZMA
void unzipLZMA(int *srcsize, unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep);
#endif

#ifdef HAVE_LZ4
void unzipLZ4(int *srcsize, unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep);
#endif

enum { kMAXZIPBUF = 0xffffff };

}

#endif
