#include "LinearFitting.cuh"

/**
   Functions related to fitting a straight line
 */

__host__ __device__ float getLineFitDistance(
  const SciFi::Tracking::LineFitterPars& parameters,
  const SciFi::SciFiHits& scifi_hits,
  const float coordX[SciFi::Tracking::max_x_hits],
  const int allXHits[SciFi::Tracking::max_x_hits],
  const int it )
{ 
  return coordX[it] - (parameters.m_c0 + (scifi_hits.z0[ allXHits[it] ] - parameters.m_z0) * parameters.m_tc);
}

__host__ __device__ float getLineFitChi2(
  const SciFi::Tracking::LineFitterPars& parameters,
  const SciFi::SciFiHits& scifi_hits,
  const float coordX[SciFi::Tracking::max_x_hits],
  const int allXHits[SciFi::Tracking::max_x_hits],
  const int it) {
  float d = getLineFitDistance( parameters, scifi_hits, coordX, allXHits, it ); 
  return d * d * coordX[it]; 
}

__host__ __device__ void solveLineFit(SciFi::Tracking::LineFitterPars &parameters)  {
  float den = (parameters.m_sz*parameters.m_sz-parameters.m_s0*parameters.m_sz2);
  parameters.m_c0  = (parameters.m_scz * parameters.m_sz - parameters.m_sc * parameters.m_sz2) / den;
  parameters.m_tc  = (parameters.m_sc *  parameters.m_sz - parameters.m_s0 * parameters.m_scz) / den;
}

__host__ __device__ void incrementLineFitParameters(
  SciFi::Tracking::LineFitterPars &parameters,
  const SciFi::SciFiHits& scifi_hits,
  const float coordX[SciFi::Tracking::max_x_hits],
  const int allXHits[SciFi::Tracking::max_x_hits],
  const int it)
{
    float c = coordX[it];
    const int hit = allXHits[it];
    float w = scifi_hits.w[hit];
    float z = scifi_hits.z0[hit] - parameters.m_z0;
    parameters.m_s0   += w;
    parameters.m_sz   += w * z;
    parameters.m_sz2  += w * z * z;
    parameters.m_sc   += w * c;
    parameters.m_scz  += w * c * z;
} 

__host__ __device__ void fitHitsFromSingleHitPlanes(
  const int it1,
  const int it2,
  const bool usedHits[SciFi::Tracking::max_x_hits],
  const SciFi::SciFiHits& scifi_hits,
  const int allXHits[SciFi::Tracking::max_x_hits],
  const int n_x_hits,
  const PlaneCounter planeCounter,
  SciFi::Tracking::LineFitterPars& lineFitParameters,
  const float coordX[SciFi::Tracking::max_x_hits],
  int otherHits[SciFi::Constants::n_layers][SciFi::Tracking::max_other_hits],
  int nOtherHits[SciFi::Constants::n_layers]) {

  for(auto itH = it1; it2 > itH; ++itH ){
    assert( itH < n_x_hits );
    if( usedHits[itH] ) continue;
    int planeCode = scifi_hits.planeCode[allXHits[itH]]/2;
    if( planeCounter.nbInPlane(planeCode) == 1 ){
      incrementLineFitParameters(lineFitParameters, scifi_hits, coordX, allXHits, itH);
    }else{
      if ( nOtherHits[planeCode] < SciFi::Tracking::max_other_hits ) {
        assert( nOtherHits[planeCode] < SciFi::Tracking::max_other_hits );
        otherHits[planeCode][ nOtherHits[planeCode]++ ] = itH;
      }
    }
  }
  solveLineFit(lineFitParameters);
}

__host__ __device__ void addAndFitHitsFromMultipleHitPlanes(
  const int nOtherHits[SciFi::Constants::n_layers],
  SciFi::Tracking::LineFitterPars& lineFitParameters,
  const SciFi::SciFiHits& scifi_hits,
  const float coordX[SciFi::Tracking::max_x_hits],
  const int allXHits[SciFi::Tracking::max_x_hits],
  const int otherHits[SciFi::Constants::n_layers][SciFi::Tracking::max_other_hits]) {

  for(int i = 0; i < SciFi::Constants::n_layers; i++){ 
    if(nOtherHits[i] == 0) continue;
    
    float bestChi2 = 1e9f;
    int best = -1;
    for( int hit = 0; hit < nOtherHits[i]; ++hit ){
      const float chi2 = getLineFitChi2(lineFitParameters, scifi_hits, coordX, allXHits, otherHits[i][hit] );
      if( chi2 < bestChi2 ){
        bestChi2 = chi2;
        best = hit;
      }
    }
    if ( best > -1 ) {
      incrementLineFitParameters(lineFitParameters, scifi_hits, coordX, allXHits,  otherHits[i][best]);
    }
  }
  solveLineFit(lineFitParameters);
}

// the track parameterization is cubic in (z-zRef),
// however only the first two parametres are varied in this fit
// -> this is a linear fit
__host__ __device__ void fastLinearFit(
  const SciFi::SciFiHits& scifi_hits,
  float trackParameters[SciFi::Tracking::nTrackParams], 
  int coordToFit[SciFi::Tracking::max_coordToFit],
  int& n_coordToFit,
  PlaneCounter planeCounter,
  SciFi::Tracking::HitSearchCuts& pars)
{
  // re-do fit every time an outlier was removed
  bool fit = true;
  while (fit) {
    //== Fit a line
    float s0   = 0.f;
    float sz   = 0.f;
    float sz2  = 0.f;
    float sd   = 0.f;
    float sdz  = 0.f;

    for ( int i_hit = 0; i_hit < n_coordToFit; ++i_hit ) {
      int hit = coordToFit[i_hit];
      const float parsX[4] = {trackParameters[0],
                              trackParameters[1],
                              trackParameters[2],
                              trackParameters[3]};
      const float zHit = scifi_hits.z0[hit];
      float track_x_at_zHit = evalCubicParameterization(parsX,zHit);
      const float d = scifi_hits.x0[hit] - track_x_at_zHit;
      const float w = scifi_hits.w[hit];
      const float z = zHit - SciFi::Tracking::zReference;
      s0   += w;
      sz   += w * z; 
      sz2  += w * z * z; 
      sd   += w * d; 
      sdz  += w * d * z; 
    }    
    float den = (sz*sz-s0*sz2);
    if( !(fabsf(den) > 1e-5f))return;
    const float da  = (sdz * sz - sd * sz2) / den; 
    const float db  = (sd *  sz - s0 * sdz) / den; 
    trackParameters[0] += da;
    trackParameters[1] += db;
    fit = false;

    if ( n_coordToFit < pars.minXHits ) return;

    int worst = n_coordToFit;
    float maxChi2 = 0.f; 
    const bool notMultiple = planeCounter.nbDifferent == n_coordToFit;
    //TODO how many multiple hits do we normaly have?
    //how often do we do the right thing here?
    //delete two hits at same time?
    for ( int i_hit = 0; i_hit < n_coordToFit; ++i_hit ) {
      int hit = coordToFit[i_hit];
      // This could certainly be wrapped in some helper function with a lot
      // of passing around or copying etc... 
      const float parsX[4] = {trackParameters[0],
                              trackParameters[1],
                              trackParameters[2],
                              trackParameters[3]};
      const float chi2 = chi2XHit( parsX, scifi_hits, hit );
      if ( chi2 > maxChi2 && ( notMultiple || planeCounter.nbInPlane( scifi_hits.planeCode[hit]/2 ) > 1 ) ) {
        maxChi2 = chi2;
        worst   = i_hit; 
      }    
    }    
    if ( maxChi2 > SciFi::Tracking::maxChi2LinearFit || ( !notMultiple && maxChi2 > 4.f ) ) {
      removeOutlier( scifi_hits, planeCounter, coordToFit, n_coordToFit, coordToFit[worst] );
      fit = true;
    }
  }
}
 
