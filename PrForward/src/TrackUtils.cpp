#include "TrackUtils.h"

std::vector<float> getTrackParameters ( float xAtRef, FullState state_at_endvelo)
{
  
  float dSlope  = ( xFromVelo(Forward::zReference,state_at_endvelo) - xAtRef ) / ( Forward::zReference - Forward::zMagnetParams[0]);
  const float zMagSlope = Forward::zMagnetParams[2] * pow(state_at_endvelo.tx,2) +  Forward::zMagnetParams[3] * pow(state_at_endvelo.ty,2);
  const float zMag    = Forward::zMagnetParams[0] + Forward::zMagnetParams[1] *  dSlope * dSlope  + zMagSlope;
  const float xMag    = xFromVelo( zMag, state_at_endvelo );
  const float slopeT  = ( xAtRef - xMag ) / ( Forward::zReference - zMag );
  dSlope        = slopeT - state_at_endvelo.tx;
  const float dyCoef  = dSlope * dSlope * state_at_endvelo.ty;
  
 std::vector<float> toreturn =  {xAtRef,
                                 slopeT,
                                 1.e-6f * Forward::xParams[0] * dSlope,
                                 1.e-9f * Forward::xParams[1] * dSlope,
                                 yFromVelo( Forward::zReference, state_at_endvelo ),
                                 state_at_endvelo.ty + dyCoef * Forward::byParams,
                                 dyCoef * Forward::cyParams,
                                 0.0,
                                 0.0 }; // last elements are chi2 and ndof, as float 
 return toreturn;
}

float calcqOverP ( float bx, FullState state_at_endvelo )
{
  
  float qop(1.0f/Gaudi::Units::GeV) ;
  float bx2  = bx * bx;
  float coef = ( Forward::momentumParams[0] +
                 Forward::momentumParams[1] * bx2 +
                 Forward::momentumParams[2] * bx2 * bx2 +
                 Forward::momentumParams[3] * bx * state_at_endvelo.tx +
                 Forward::momentumParams[4] * pow(state_at_endvelo.ty,2) +
                 Forward::momentumParams[5] * pow(state_at_endvelo.ty,2) * pow(state_at_endvelo.ty,2) );
  float m_slope2 = pow(state_at_endvelo.tx,2) + pow(state_at_endvelo.ty,2);
  float proj = sqrt( ( 1.f + m_slope2 ) / ( 1.f + pow(state_at_endvelo.tx,2) ) ); 
  qop = ( state_at_endvelo.tx - bx ) / ( coef * Gaudi::Units::GeV * proj * Forward::magscalefactor) ;
  return qop ;
  
}

// DvB: what does this do?
// -> get position within magnet (?)
float zMagnet(FullState state_at_endvelo)
{
    
  return ( Forward::zMagnetParams[0] +
           Forward::zMagnetParams[2] * pow(state_at_endvelo.tx,2) +
           Forward::zMagnetParams[3] * pow(state_at_endvelo.ty,2) );
}

void covariance ( FullState& state, const float qOverP )
{
     
  state.c00 = Forward::covarianceValues[0];
  state.c11 = Forward::covarianceValues[1];
  state.c22 = Forward::covarianceValues[2];
  state.c33 = Forward::covarianceValues[3];
  state.c44 = Forward::covarianceValues[4] * qOverP * qOverP;
}

float calcDxRef(float pt, FullState state_at_endvelo) {
  float m_slope2 = pow(state_at_endvelo.tx,2) + pow(state_at_endvelo.ty,2);
  return 3973000. * sqrt( m_slope2 ) / pt - 2200. *  pow(state_at_endvelo.ty,2) - 1000. * pow(state_at_endvelo.tx,2); // tune this window
}

float trackToHitDistance( std::vector<float> trackParameters, ForwardTracking::HitsSoAFwd* hits_layers, int hit )
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




