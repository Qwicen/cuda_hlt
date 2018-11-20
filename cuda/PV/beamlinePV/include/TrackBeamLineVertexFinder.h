#pragma once

#include "Common.h"
#include "VeloConsolidated.cuh" 
#include "PV_Definitions.cuh"
#include "VeloConsolidated.cuh"
#include <math.h>
#include "float.h"
#include <algorithm>


unsigned int m_minNumTracksPerVertex = 5;
float        m_zmin = - 300.; //unit: mm Min z position of vertex seed
float        m_zmax = 300; //unit: mm Max z position of vertex seed
float        m_dz = 0.25; //unit: mm Z histogram bin size
float        m_maxTrackZ0Err = 1.5; // unit: mm "Maximum z0-error for adding track to histo"
float        m_minDensity = 1.0; // unit: 1./mm "Minimal density at cluster peak  (inverse resolution)"
float        m_minDipDensity = 2.0; // unit: 1./mm,"Minimal depth of a dip to split cluster (inverse resolution)"
float        m_minTracksInSeed = 2.5; // "MinTrackIntegralInSeed"
float        m_maxVertexRho = 0.3; // unit:: mm "Maximum distance of vertex to beam line" 
unsigned int m_maxFitIter = 5; // "Maximum number of iterations for vertex fit"
float        m_maxDeltaChi2 = 9; //"Maximum chi2 contribution of track to vertex fit"



// structure with minimal track info needed for PV search
struct PVTrack
{
  PVTrack() {}
PVTrack( const Velo::State& state, double dz, unsigned short _index )
: z{float(state.z+dz)},
    x{float(state.x+dz*state.tx),float(state.y+dz*state.ty)},
    tx{float(state.tx),float(state.ty)},index{_index}
  {
    // perhaps we should invert it /before/ switching to single FPP?
    // it doesn't seem to make much difference.
    
    PV::myfloat state_tmp_c00 = state.c00;
    PV::myfloat state_tmp_c11 = state.c11;
    
    double dz2 = dz*dz ;
    
    state_tmp_c00 += dz2 * state.c22 + 2*dz* state.c20 ;
    state_tmp_c11 += dz2* state.c33 + 2* dz*state.c31 ;
    W_00 = 1. / state_tmp_c00;
    W_11 = 1. / state_tmp_c11;
  }
  float z{0} ;
  float2 x ;      /// position (x,y)
  float2 tx ;     /// direction (tx,ty)
  // to do: check whether this needs to be a double
  double W_00 ; /// weightmatrix
  double W_11 ;
  unsigned short index{0} ;/// index in the list with tracks
} ;

template<typename FTYPE> FTYPE sqr( FTYPE x ) { return x*x ;} 

struct Extremum
{
Extremum( unsigned short _index, float _value, float _integral ) :
  index{_index}, value{_value}, integral{_integral} {}
  unsigned short index;
  float value ;
  float integral ;
} ;

struct Cluster
{
Cluster( unsigned short _izfirst, unsigned short _izlast,  unsigned short _izmax ) :
  izfirst{_izfirst}, izlast{_izlast}, izmax{_izmax} {}
  unsigned short izfirst ;
  unsigned short izlast ;
  unsigned short izmax ;
} ;

struct SeedZWithIteratorPair
{
  using iterator = std::vector< PVTrack >::iterator  ;
  float z ;
  iterator begin ;
  iterator end ;
SeedZWithIteratorPair( float _z, iterator _begin, iterator _end) :
  z{_z},begin{_begin},end{_end} {}
} ;

// Need a small extension to the track when fitting the
// vertex. Caching this information doesn't seem to help much
// though.
struct PVTrackInVertex : PVTrack
{
 PVTrackInVertex( const PVTrack& trk )
   : PVTrack{trk}
  {
    //H matrix is symmetric and has four non-zero entries
    H_00 = 1. ;
    H_11 = 1. ;
    H_20 = - trk.tx.x ;
    H_21 = - trk.tx.y ;
    //HW: product of H and W matrices, symmetric with four non-zero entries
    HW_00 = W_00;
    HW_11 = W_11;
    HW_20 = H_20 * W_00;
    HW_21 = H_21 * W_11;
    HWH_00 = W_00;
    HWH_20 = H_20 * W_00;
    HWH_11 = W_11;
    HWH_21 = H_21 * W_11;
    HWH_22 = H_20*H_20*W_00 + H_21*H_21*W_11;
  }
  // TO DO: check whether this needs to be a double
  double H_00 ;
  double H_11 ;
  double H_20 ;
  double H_21 ;
  //HW: product of H and W matrices, symmetric with four non-zero entries
  double HW_00 ;
  double HW_11 ;
  double HW_20 ;
  double HW_21 ;
  double HWH_00 ;
  double HWH_20 ;
  double HWH_11 ;
  double HWH_21 ;
  double HWH_22 ;
  float weight{1} ;
} ;

