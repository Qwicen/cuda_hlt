#include "HoughTransform.cuh"

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
  const float xFromVelo_Hit = straightLineExtend(xParams_seed,zHit);
  const float ty2 = velo_state.ty*velo_state.ty;
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

__host__ __device__ int fitParabola(
  int* coordToFit,
  const int n_coordToFit,
  const SciFi::SciFiHits& scifi_hits,
  float trackParameters[SciFi::Tracking::nTrackParams],
  const bool xFit ) {

  //== Fit a cubic
  float s0   = 0.f; 
  float sz   = 0.f; 
  float sz2  = 0.f; 
  float sz3  = 0.f; 
  float sz4  = 0.f; 
  float sd   = 0.f; 
  float sdz  = 0.f; 
  float sdz2 = 0.f; 
  
  for ( int i_hit = 0; i_hit < n_coordToFit; ++i_hit) {
    int hit = coordToFit[i_hit];
    float d = trackToHitDistance(trackParameters, scifi_hits, hit);
    if (!xFit)
      d *= - 1. / scifi_hits.dxdy[hit];//TODO multiplication much faster than division!
    float w = scifi_hits.w[hit];
    float z = .001f * ( scifi_hits.z0[hit] - SciFi::Tracking::zReference );
    s0   += w;
    sz   += w * z; 
    sz2  += w * z * z; 
    sz3  += w * z * z * z; 
    sz4  += w * z * z * z * z; 
    sd   += w * d; 
    sdz  += w * d * z; 
    sdz2 += w * d * z * z; 
  }    
  const float b1 = sz  * sz  - s0  * sz2; 
  const float c1 = sz2 * sz  - s0  * sz3; 
  const float d1 = sd  * sz  - s0  * sdz; 
  const float b2 = sz2 * sz2 - sz * sz3; 
  const float c2 = sz3 * sz2 - sz * sz4; 
  const float d2 = sdz * sz2 - sz * sdz2;
  const float den = (b1 * c2 - b2 * c1 );
  if(!(fabsf(den) > 1e-5)) return false;
  const float db  = (d1 * c2 - d2 * c1 ) / den; 
  const float dc  = (d2 * b1 - d1 * b2 ) / den; 
  const float da  = ( sd - db * sz - dc * sz2) / s0;
  if (xFit) {
    trackParameters[0] += da;
    trackParameters[1] += db*1.e-3f;
    trackParameters[2] += dc*1.e-6f;
  } else {
    trackParameters[4] += da;
    trackParameters[5] += db*1.e-3f;
    trackParameters[6] += dc*1.e-6f;
  }

  return true;
}

__host__ __device__ bool fitXProjection(
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
    //int   nDoF = -3; // fitted 3 parameters
    int  nDoF = -3;
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
  const float wMag   = 1./(tolYMag * tolYMag );

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
