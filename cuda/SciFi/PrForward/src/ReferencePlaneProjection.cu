#include "ReferencePlaneProjection.cuh"

// project x position of hits from one plane onto the reference plane
// save it in coordX
//in the c++ this is vectorized, undoing because no point before CUDA (but vectorization is obvious)
__host__ __device__ void xAtRef_SamePlaneHits(
  const SciFi::SciFiHits& scifi_hits,
  const int allXHits[SciFi::Tracking::max_x_hits],
  const int n_x_hits,
  float coordX[SciFi::Tracking::max_x_hits],
  const float xParams_seed[4],
  SciFi::Tracking::Arrays* constArrays,
  MiniState velo_state,
  const float zMagSlope, 
  int itH, int itEnd)
{
  //this is quite computationally expensive mind, should take care when porting
  assert( itH < SciFi::Tracking::max_x_hits );
  const float zHit    = scifi_hits.z0[allXHits[itH]]; //all hits in same layer
  const float xFromVelo_Hit = evalCubicParameterization(xParams_seed,zHit);
  const float dSlopeDivPart = 1.f / ( zHit - constArrays->zMagnetParams[0]);
  const float dz      = 1.e-3f * ( zHit - SciFi::Tracking::zReference );
  
  while( itEnd>itH ){
    float xHit = scifi_hits.x0[allXHits[itH]];
    // difference in slope before and after the kick
    float dSlope  = ( xFromVelo_Hit - xHit ) * dSlopeDivPart;
    // update zMag now that dSlope is known
    float zMag = zMagSlope + constArrays->zMagnetParams[1] *  dSlope * dSlope;
    float xMag    = xFromVelo_Hit + velo_state.tx * (zMag - zHit);
    // calculate x position on reference plane (save in coodX)
    // dxCoef: account for additional bending of track due to fringe field in first station
    // expressed by quadratic and cubic term in z
    float dxCoef  = dz * dz * ( constArrays->xParams[0] + dz * constArrays->xParams[1] ) * dSlope;
    float ratio   = (  SciFi::Tracking::zReference - zMag ) / ( zHit - zMag );
    coordX[itH] = xMag + ratio * (xHit + dxCoef  - xMag);
    itH++;
  }
}
