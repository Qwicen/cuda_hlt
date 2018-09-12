#pragma once

/**
   Helper functions related to properties of hits on planes
 */

// check that val is within [min, max]
inline bool isInside(float val, const float min, const float max) {
  return (val > min) && (val < max) ;
}

// get lowest index where range[index] > value, within [start,end] of range 
inline int getLowerBound(float range[],float value,int start, int end) {
  int i = start;
  for (; i<end; i++) {
    if (range[i] > value) break;
  }
  return i;
}

// count number of planes with more than 0 hits
inline int nbDifferent(int planelist[]) {
  int different = 0;
  for (int i=0;i<12;++i){different += planelist[i] > 0 ? 1 : 0;}
  return different;
}

// count number of planes with a single hit
inline int nbSingle(int planelist[]) {
  int single = 0;
  for (int i=0;i<12;++i){single += planelist[i] == 1 ? 1 : 0;}
  return single;
}

struct PlaneCounter{
  int planeList[SciFi::Constants::n_physical_layers];
  unsigned int nbDifferent;

  inline void addHit( int plane ) {
    nbDifferent += (int)( (planeList[plane] += 1 ) == 1) ;
  }

  inline void removeHit( int plane ) {
    nbDifferent -= ((int)( (planeList[plane] -= 1 ) == 0)) ;
  }

  inline int nbInPlane( int plane ) {
    return planeList[plane];
  }
  
  inline void clear() {
    nbDifferent = 0;
    for ( int i = 0; i < SciFi::Constants::n_physical_layers; ++i ) {
      planeList[i] = 0;
    }
  }
  
};
