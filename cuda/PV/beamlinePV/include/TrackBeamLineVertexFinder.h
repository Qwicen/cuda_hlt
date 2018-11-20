#pragma once 

#include "Common.h"
#include "VeloConsolidated.cuh" 
#include "PV_Definitions.cuh"
#include "float_operations.h"

#include <math.h>
#include <algorithm>


// structure with minimal track info needed for PV search
struct PVTrack
{
  __host__ __device__ PVTrack() {}
__host__ __device__ PVTrack( const Velo::State& state, double dz, unsigned short _index )
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

template<typename FTYPE> 
__host__ __device__ FTYPE sqr( FTYPE x ) { return x*x ;} 

struct Extremum
{
__host__ __device__ Extremum( unsigned short _index, float _value, float _integral ) :
  index{_index}, value{_value}, integral{_integral} {}
  unsigned short index;
  float value ;
  float integral ;
} ;

struct Cluster
{
__host__ __device__ Cluster( unsigned short _izfirst, unsigned short _izlast,  unsigned short _izmax ) :
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
__host__ __device__ SeedZWithIteratorPair( float _z, iterator _begin, iterator _end) :
  z{_z},begin{_begin},end{_end} {}
} ;

// Need a small extension to the track when fitting the
// vertex. Caching this information doesn't seem to help much
// though.
struct PVTrackInVertex : PVTrack
{
__host__ __device__  PVTrackInVertex( const PVTrack& trk )
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

std::vector<PV::Vertex> findPVs(
  uint* kalmanvelo_states,
  int * velo_atomics,
  uint* velo_track_hit_number,
  const uint number_of_events 
); 

