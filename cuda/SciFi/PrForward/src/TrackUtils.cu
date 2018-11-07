#include "TrackUtils.cuh"

__host__ __device__ void getTrackParameters (
  float xAtRef,
  MiniState velo_state,
  SciFi::Tracking::Arrays* constArrays,
  float trackParams[SciFi::Tracking::nTrackParams])
{
  
  float dSlope  = ( xFromVelo(SciFi::Tracking::zReference,velo_state) - xAtRef ) / ( SciFi::Tracking::zReference - constArrays->zMagnetParams[0]);
  const float zMagSlope = constArrays->zMagnetParams[2] * velo_state.tx*velo_state.tx +  constArrays->zMagnetParams[3] * velo_state.ty*velo_state.ty;
  const float zMag    = constArrays->zMagnetParams[0] + constArrays->zMagnetParams[1] *  dSlope * dSlope  + zMagSlope;
  const float xMag    = xFromVelo( zMag, velo_state );
  const float slopeT  = ( xAtRef - xMag ) / ( SciFi::Tracking::zReference - zMag );
  dSlope        = slopeT - velo_state.tx;
  const float dyCoef  = dSlope * dSlope * velo_state.ty;
  
  trackParams[0] = xAtRef;
  trackParams[1] = slopeT;
  trackParams[2] = 1.e-6f * constArrays->xParams[0] * dSlope;
  trackParams[3] = 1.e-9f * constArrays->xParams[1] * dSlope;
  trackParams[4] = yFromVelo( SciFi::Tracking::zReference, velo_state );
  trackParams[5] = velo_state.ty + dyCoef * SciFi::Tracking::byParams;
  trackParams[6] = dyCoef * SciFi::Tracking::cyParams;
  trackParams[7] = 0.0f;
  trackParams[8] = 0.0f; // last elements are chi2 and ndof, as float 
}

__host__ __device__ float calcqOverP (
  float bx,
  SciFi::Tracking::Arrays* constArrays,
  MiniState velo_state )
{
  
  float qop(1.0f/Gaudi::Units::GeV) ;
  const float bx2  = bx * bx;
  const float ty2 = velo_state.ty*velo_state.ty;
  const float coef = ( constArrays->momentumParams[0] +
                 constArrays->momentumParams[1] * bx2 +
                 constArrays->momentumParams[2] * bx2 * bx2 +
                 constArrays->momentumParams[3] * bx * velo_state.tx +
                 constArrays->momentumParams[4] * ty2 +
                 constArrays->momentumParams[5] * ty2 * ty2 );
  const float tx2 = velo_state.tx*velo_state.tx;
  float m_slope2 = tx2 + ty2;
  float proj = sqrtf( ( 1.f + m_slope2 ) / ( 1.f + tx2 ) ); 
  qop = ( velo_state.tx - bx ) / ( coef * Gaudi::Units::GeV * proj * SciFi::Tracking::magscalefactor) ;
  return qop ;
  
}

// Find z zMag position within the magnet at which the bending ("kick") occurs
// this is parameterized based on MC
// the second parameter([1]) is multiplied by the difference in slope before and
// after the kick, this slope is calculated from zMag and the x position of the track
// at the reference plane -> it is calculated iteratively later
__host__ __device__ float zMagnet(
  MiniState velo_state,
  SciFi::Tracking::Arrays* constArrays)
{
    
  return ( constArrays->zMagnetParams[0] +
           constArrays->zMagnetParams[2] * velo_state.tx*velo_state.tx +
           constArrays->zMagnetParams[3] * velo_state.ty*velo_state.ty );
}

__host__ __device__ void covariance (
  FullState& state,
  SciFi::Tracking::Arrays* constArrays,
  const float qOverP )
{
     
  state.c00 = constArrays->covarianceValues[0];
  state.c11 = constArrays->covarianceValues[1];
  state.c22 = constArrays->covarianceValues[2];
  state.c33 = constArrays->covarianceValues[3];
  state.c44 = constArrays->covarianceValues[4] * qOverP * qOverP;
}

// calculate difference between straight line extrapolation and
// where a track with wrongSignPT (2 GeV) would be on the reference plane (?)
__host__ __device__ float calcDxRef(float pt, MiniState velo_state) {
  const float tx2 = velo_state.tx*velo_state.tx;
  const float ty2 = velo_state.ty*velo_state.ty;
  float m_slope2 = tx2 + ty2;
  return 3973000.f * sqrtf( m_slope2 ) / pt - 2200.f *  ty2 - 1000.f * tx2; // tune this window
}

__host__ __device__ float trackToHitDistance(
  float trackParameters[SciFi::Tracking::nTrackParams],
  const SciFi::SciFiHits& scifi_hits,
  int hit )
{
  const float parsX[4] = {trackParameters[0],
                          trackParameters[1],
                          trackParameters[2],
                          trackParameters[3]};
  const float parsY[4] = {trackParameters[4],
                          trackParameters[5],
                          trackParameters[6],
                          0.f}; 
  float z_Hit = scifi_hits.z0[hit] + 
    scifi_hits.dzdy[hit]*evalCubicParameterization(parsY, scifi_hits.z0[hit]);
  float x_track = evalCubicParameterization(parsX,z_Hit);
  float y_track = evalCubicParameterization(parsY,z_Hit);
  return scifi_hits.x0[hit] + y_track*scifi_hits.dxdy[hit] - x_track; 
}

__host__ __device__ float chi2XHit(
  const float parsX[4],
  const SciFi::SciFiHits& scifi_hits,
  const int hit ) {
  float track_x_at_zHit = evalCubicParameterization(parsX,scifi_hits.z0[hit]);
   float hitdist = scifi_hits.x0[hit] - track_x_at_zHit; 
   return hitdist*hitdist*scifi_hits.w[hit];
}

// the track parameterization is cubic in (z-zRef),
// however only the first three parametres are varied in this fit only
// -> this is a quadratic fit
__host__ __device__ bool quadraticFitX(
  const SciFi::SciFiHits& scifi_hits,
  float trackParameters[SciFi::Tracking::nTrackParams],
  int coordToFit[SciFi::Tracking::max_coordToFit],
  int& n_coordToFit,
  PlaneCounter& planeCounter,
  SciFi::Tracking::HitSearchCuts& pars)
{

  if (planeCounter.nbDifferent < pars.minXHits) return false;
  bool doFit = true;
  while ( doFit ) {

    fitParabola( coordToFit, n_coordToFit, scifi_hits, trackParameters, true );
    
    float maxChi2 = 0.f; 
    float totChi2 = 0.f;  
    int   nDoF = -3; // fitted 3 parameters
    const bool notMultiple = planeCounter.nbDifferent == n_coordToFit;

    int worst = n_coordToFit;
    for ( int i_hit = 0; i_hit < n_coordToFit; ++i_hit ) {
      int hit = coordToFit[i_hit];
      float d = trackToHitDistance(trackParameters, scifi_hits, hit);
      float chi2 = d*d*scifi_hits.w[hit];
      totChi2 += chi2;
      ++nDoF;
      if ( chi2 > maxChi2 && ( notMultiple || planeCounter.nbInPlane( scifi_hits.planeCode[hit]/2 ) > 1 ) ) {
        maxChi2 = chi2;
        worst   = i_hit; 
      }    
    }    
    if ( nDoF < 1 )return false;
    trackParameters[7] = totChi2;
    trackParameters[8] = (float) nDoF;

    if ( worst == n_coordToFit ) {
      return true;
    }    
    doFit = false;
    if ( totChi2/nDoF > SciFi::Tracking::maxChi2PerDoF  ||
         maxChi2 > SciFi::Tracking::maxChi2XProjection ) {
      removeOutlier( scifi_hits, planeCounter, coordToFit, n_coordToFit, coordToFit[worst]);
      if (planeCounter.nbDifferent < pars.minXHits + pars.minStereoHits) return false;
      doFit = true;
    }    
  }
  return true;
}
 

__host__ __device__ bool fitYProjection(
  const SciFi::SciFiHits& scifi_hits,
  SciFi::Tracking::Track& track,
  int stereoHits[SciFi::Tracking::max_stereo_hits],
  int& n_stereoHits,
  PlaneCounter& planeCounter,
  MiniState velo_state,
  SciFi::Tracking::Arrays* constArrays,
  SciFi::Tracking::HitSearchCuts& pars)
{
  
  float maxChi2 = 1.e9f;
  bool parabola = false; //first linear than parabola
  //== Fit a line
  const float txs  = track.trackParams[0]; // simplify overgeneral c++ calculation
  const float tsxz = velo_state.x + (SciFi::Tracking::zReference - velo_state.z) * velo_state.tx; 
  const float tolYMag = SciFi::Tracking::tolYMag + SciFi::Tracking::tolYMagSlope * fabsf(txs-tsxz);
  const float wMag   = 1.f/(tolYMag * tolYMag );

  bool doFit = true;
  while ( doFit ) {
    //Use position in magnet as constrain in fit
    //although because wMag is quite small only little influence...
    float zMag  = zMagnet(velo_state, constArrays);
    const float tys = track.trackParams[4]+(zMag-SciFi::Tracking::zReference)*track.trackParams[5];
    const float tsyz = velo_state.y + (zMag-velo_state.z)*velo_state.ty;
    const float dyMag = tys-tsyz;
    zMag -= SciFi::Tracking::zReference;
    float s0   = wMag;
    float sz   = wMag * zMag;
    float sz2  = wMag * zMag * zMag;
    float sd   = wMag * dyMag;
    float sdz  = wMag * dyMag * zMag;
   
    if ( parabola ) {

      // position in magnet not used for parabola fit, hardly any influence on efficiency
      fitParabola( stereoHits, n_stereoHits, scifi_hits, track.trackParams, false );
      
    } else { // straight line fit

      for ( int i_hit = 0; i_hit < n_stereoHits; ++i_hit ) {
        int hit = stereoHits[i_hit];
        const float d = - trackToHitDistance(track.trackParams, scifi_hits, hit) / 
                          scifi_hits.dxdy[hit];//TODO multiplication much faster than division!
        const float w = scifi_hits.w[hit];
        const float z = scifi_hits.z0[hit] - SciFi::Tracking::zReference;
	s0   += w;
        sz   += w * z; 
        sz2  += w * z * z;
        sd   += w * d;
        sdz  += w * d * z;
      }
      const float den = (s0 * sz2 - sz * sz );
      if(!(fabsf(den) > 1e-5)) { 
        return false;
      }
      const float da  = (sd * sz2 - sdz * sz ) / den;
      const float db  = (sdz * s0 - sd  * sz ) / den;
      track.trackParams[4] += da;
      track.trackParams[5] += db;
    }//fit end, now doing outlier removal

    int worst = n_stereoHits;
    maxChi2 = 0.;
    for ( int i_hit = 0; i_hit < n_stereoHits; ++i_hit ) {
      int hit = stereoHits[i_hit];
      float d = trackToHitDistance(track.trackParams, scifi_hits, hit);
      float chi2 = d*d*scifi_hits.w[hit];
      if ( chi2 > maxChi2 ) {
        maxChi2 = chi2;
        worst   = i_hit;
      }
    }

    if ( maxChi2 < SciFi::Tracking::maxChi2StereoLinear && !parabola ) {
      parabola = true;
      maxChi2 = 1.e9f;
      continue;
    }

    if ( maxChi2 > SciFi::Tracking::maxChi2Stereo ) {
      removeOutlier( scifi_hits, planeCounter, stereoHits, n_stereoHits, stereoHits[worst] );
      if ( planeCounter.nbDifferent < pars.minStereoHits ) {
        return false;
      }
      continue;
    }
    break;
  }
  return true;
} 
 
