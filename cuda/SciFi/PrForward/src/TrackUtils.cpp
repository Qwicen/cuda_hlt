#include "TrackUtils.h"

__host__ __device__ void getTrackParameters ( float xAtRef, MiniState velo_state, float trackParams[SciFi::Tracking::nTrackParams])
{
  
  float dSlope  = ( xFromVelo(SciFi::Tracking::zReference,velo_state) - xAtRef ) / ( SciFi::Tracking::zReference - SciFi::Tracking::zMagnetParams[0]);
  const float zMagSlope = SciFi::Tracking::zMagnetParams[2] * pow(velo_state.tx,2) +  SciFi::Tracking::zMagnetParams[3] * pow(velo_state.ty,2);
  const float zMag    = SciFi::Tracking::zMagnetParams[0] + SciFi::Tracking::zMagnetParams[1] *  dSlope * dSlope  + zMagSlope;
  const float xMag    = xFromVelo( zMag, velo_state );
  const float slopeT  = ( xAtRef - xMag ) / ( SciFi::Tracking::zReference - zMag );
  dSlope        = slopeT - velo_state.tx;
  const float dyCoef  = dSlope * dSlope * velo_state.ty;
  
  trackParams[0] = xAtRef;
  trackParams[1] = slopeT;
  trackParams[2] = 1.e-6f * SciFi::Tracking::xParams[0] * dSlope;
  trackParams[3] = 1.e-9f * SciFi::Tracking::xParams[1] * dSlope;
  trackParams[4] = yFromVelo( SciFi::Tracking::zReference, velo_state );
  trackParams[5] = velo_state.ty + dyCoef * SciFi::Tracking::byParams;
  trackParams[6] = dyCoef * SciFi::Tracking::cyParams;
  trackParams[7] = 0.0;
  trackParams[8] = 0.0; // last elements are chi2 and ndof, as float 
}

__host__ __device__ float calcqOverP ( float bx, MiniState velo_state )
{
  
  float qop(1.0f/Gaudi::Units::GeV) ;
  float bx2  = bx * bx;
  float coef = ( SciFi::Tracking::momentumParams[0] +
                 SciFi::Tracking::momentumParams[1] * bx2 +
                 SciFi::Tracking::momentumParams[2] * bx2 * bx2 +
                 SciFi::Tracking::momentumParams[3] * bx * velo_state.tx +
                 SciFi::Tracking::momentumParams[4] * pow(velo_state.ty,2) +
                 SciFi::Tracking::momentumParams[5] * pow(velo_state.ty,2) * pow(velo_state.ty,2) );
  float m_slope2 = pow(velo_state.tx,2) + pow(velo_state.ty,2);
  float proj = sqrt( ( 1.f + m_slope2 ) / ( 1.f + pow(velo_state.tx,2) ) ); 
  qop = ( velo_state.tx - bx ) / ( coef * Gaudi::Units::GeV * proj * SciFi::Tracking::magscalefactor) ;
  return qop ;
  
}

// DvB: what does this do?
// -> get position within magnet (?)
__host__ __device__ float zMagnet(MiniState velo_state)
{
    
  return ( SciFi::Tracking::zMagnetParams[0] +
           SciFi::Tracking::zMagnetParams[2] * pow(velo_state.tx,2) +
           SciFi::Tracking::zMagnetParams[3] * pow(velo_state.ty,2) );
}

__host__ __device__ void covariance ( FullState& state, const float qOverP )
{
     
  state.c00 = SciFi::Tracking::covarianceValues[0];
  state.c11 = SciFi::Tracking::covarianceValues[1];
  state.c22 = SciFi::Tracking::covarianceValues[2];
  state.c33 = SciFi::Tracking::covarianceValues[3];
  state.c44 = SciFi::Tracking::covarianceValues[4] * qOverP * qOverP;
}

// calculate difference between straight line extrapolation and
// where a track with wrongSignPT (2 GeV) would be on the reference plane (?)
__host__ __device__ float calcDxRef(float pt, MiniState velo_state) {
  float m_slope2 = pow(velo_state.tx,2) + pow(velo_state.ty,2);
  return 3973000. * sqrt( m_slope2 ) / pt - 2200. *  pow(velo_state.ty,2) - 1000. * pow(velo_state.tx,2); // tune this window
}

__host__ __device__ float trackToHitDistance( float trackParameters[SciFi::Tracking::nTrackParams], SciFi::HitsSoA* hits_layers, int hit )
{
  const float parsX[4] = {trackParameters[0],
                          trackParameters[1],
                          trackParameters[2],
                          trackParameters[3]};
  const float parsY[4] = {trackParameters[4],
                          trackParameters[5],
                          trackParameters[6],
                          0.}; 
  float z_Hit = hits_layers->m_z[hit] + 
    hits_layers->m_dzdy[hit]*straightLineExtend(parsY, hits_layers->m_z[hit]);
  float x_track = straightLineExtend(parsX,z_Hit);
  float y_track = straightLineExtend(parsY,z_Hit);
  return hits_layers->m_x[hit] + y_track*hits_layers->m_dxdy[hit] - x_track; 
}

__host__ __device__ float chi2XHit( const float parsX[4], SciFi::HitsSoA* hits_layers, const int hit ) {
  float track_x_at_zHit = straightLineExtend(parsX,hits_layers->m_z[hit]);
   float hitdist = hits_layers->m_x[hit] - track_x_at_zHit; 
   return hitdist*hitdist*hits_layers->m_w[hit];
}


