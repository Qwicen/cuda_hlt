#include "ParKalmanMethods.cuh"

using namespace ParKalmanFilter;

typedef Vector<10> Vector10;
typedef Vector<2> Vector2;
typedef SquareMatrix<true,2> SymMatrix2x2;
typedef SquareMatrix<false,2> Matrix2x2;
  
  //----------------------------------------------------------------------
  // Create a VELO seed state.
  __device__ void CreateVeloSeedState(
    const Velo::Consolidated::Hits &hits,
    const int nVeloHits,
    int nHit,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  ) {

    // Set the state.
    x(0) = (double)hits.x[nVeloHits-1-nHit];
    x(1) = (double)hits.y[nVeloHits-1-nHit];
    x(2) = (double)((hits.x[0]-hits.x[nVeloHits-1-nHit])/(hits.z[0]-hits.z[nVeloHits-1-nHit]));
    x(3) = (double)((hits.y[0]-hits.y[nVeloHits-1-nHit])/(hits.z[0]-hits.z[nVeloHits-1-nHit]));
    x(4) = tI.m_BestMomEst;
    lastz = (double)hits.z[nVeloHits-1-nHit];
    
    // Set covariance matrix with large uncertainties and no correlations.
    C(0,0)=100; C(0,1)=0; C(0,2)=0; C(0,3)=0;   C(0,4)=0;
    C(1,1)=100; C(1,2)=0; C(1,3)=0; C(1,4)=0;
    C(2,2)=1;   C(2,3)=0; C(2,4)=0;
    C(3,3)=1;   C(3,4)=0;
    C(4,4)=0.09*x(4)*x(4);

  }

  //----------------------------------------------------------------------
  // Predict inside the VELO.
  __device__ void PredictStateV(
    const Velo::Consolidated::Hits &hits,
    int nHit,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  ) {
    
    // Transportation and noise.
    Matrix5x5 F;
    SymMatrix5x5 Q;
    //x[0] = tI.m_extr->Par_predictV[0][5];
    ExtrapolateInV(lastz, (double)hits.z[nHit], x, F, Q, tI);

    // Transport the covariance matrix.
    C = similarity_5_5(F,C);

    // Add noise.
    //C += Q;
    C = C + Q;
    
    // Set current z position.
    lastz = (double)hits.z[nHit];
  }

  //----------------------------------------------------------------------
  // Predict VELO <-> UT
  __device__ bool PredictStateVUT(
    const Velo::Consolidated::Hits &hitsVelo,
    const UT::Consolidated::Hits &hitsUT, // These probably don't exist yet.
    const int nVeloHits,
    const int nUTHits,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  ) {
    
    int forward = lastz<1000. ? 1 : -1;
    bool returnVal = true;
    Vector5 xOld = x;

    // Predicted z position.
    double z;
    // Noise.
    SymMatrix5x5 Q;
    // Jacobian.
    Matrix5x5 F;

    // Prediction. For now only works if the track has UT hits.
    if(forward>0){
      tI.m_RefStateForwardV=x;

      // Extrapolate to first layer if there's a hit.
      if(tI.m_UTLayerIdxs[0]>=0){
        z = (double)hitsUT.zAtYEq0[tI.m_UTLayerIdxs[0]];
        returnVal = ExtrapolateVUT(lastz, z, x, F, Q, tI);
      }
      // Otherwise extrapolate to end of VUT region.
      else{
        z = tI.m_extr->VUTExtrEndZ();
        returnVal = ExtrapolateVUT(lastz, z, x, F, Q, tI);
      }

      C = similarity_5_5(F, C);
      
      C = C + Q;
      tI.m_RefPropForwardVUT = F;
      tI.m_RefStateForwardFUT = x;
    }
    // T to VELO prediction (using forward as reference).
    else{
      //z = (double)hitsVelo.z[nVeloHits-1];
      z = (double)hitsVelo.z[0]; // Velo hits are stored backward??
      F = tI.m_RefPropForwardVUT;
      F = inverse(F);
      x = tI.m_RefStateForwardV + F*(x-tI.m_RefStateForwardFUT);
      C = similarity_5_5(F, C);
      GetNoiseVUTBackw(lastz, z, xOld, Q, tI);
      C = C + Q;
    }
    // TODO: Do we need a prediction without reference?
    
    // Set current z position.
    lastz = z;
    return returnVal;
    
  }

  //----------------------------------------------------------------------
  // Predict UT <-> UT.
  __device__ void PredictStateUT(
    const UT::Consolidated::Hits &hits,
    const uint layer,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  ) {
    double z;
    Matrix5x5 F;
    SymMatrix5x5 Q;

    // Check if there's a hit in this layer and extrapolate to it.
    if(tI.m_UTLayerIdxs[layer]>=0){
      z = (double)hits.zAtYEq0[tI.m_UTLayerIdxs[layer]];
      ExtrapolateInUT(lastz, layer, z, x, F, Q, tI);
    }
    // Otherwise add noise without changing the state.
    else{
      z = lastz;
      ExtrapolateInUT(lastz, layer, z, x, F, Q, tI);
    }

    C = similarity_5_5(F, C);
        
    C = C + Q;
    lastz = z;
  }

  //----------------------------------------------------------------------
  // Predict UT <-> T precise version (what does that mean?)
  __device__ void PredictStateUTT(
    const UT::Consolidated::Hits &hits,
    const int n_ut_hits,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI) {
    
    int forward = lastz<5000. ? 1 : -1;
    Vector5 xOld = x;

    // When going backward predict to fixed z in T(z=7855).
    if(forward<0){
      PredictStateTFT(forward, x, C, lastz, tI);
    }
    else if(tI.m_extr->UTTExtrBeginZ() != lastz){
      PredictStateUTFUT(forward, x, C, lastz, tI);
    }

    // Jacobian.
    Matrix5x5 F;

    // Extrapolating from last UT hit (if no hit: z is set to a
    // default value) to fixed z in T (z position is a parameter set
    // to the middle of the first layer).
    if(forward>0){
      if(tI.m_extr->UTTExtrBeginZ() != lastz){} // Does nothing?
      
      // Calculate the extrapolation for a reference state that uses
      // the initial forward momentum estimate.      
      Vector5 xref = x;
      // Switch out for the best momentum measurement.
      xref[4] = tI.m_BestMomEst;
      tI.m_RefStateForwardUT = xref;

      // Transportation and noise matrices.
      Matrix5x5 F;
      SymMatrix5x5 Q;
      ExtrapolateUTT(xref, F, Q, tI);

      // Save reference state/jacobian after this intermediate extrapolation.
      tI.m_RefStateForwardT = xref;
      tI.m_RefPropForwardUTT = F;
      x = tI.m_RefStateForwardT + F*(x-tI.m_RefStateForwardUT);
      lastz = tI.m_extr->UTTExtrEndZ();
      C = similarity_5_5(F, C);
      C = C + Q;
    }
    // No parametrization for this -> use reference (what does this mean?)
    else{
      F = tI.m_RefPropForwardUTT;
      F = inverse(F);
      x = tI.m_RefStateForwardUT + F*(xOld - tI.m_RefStateForwardT);
      lastz = tI.m_extr->UTTExtrBeginZ();
      C = similarity_5_5(F, C);
      SymMatrix5x5 Q;
      GetNoiseUTTBackw(xOld, Q, tI);
      C = C + Q;
    }

    // When going backwards: predict to the last VELO measurement.
    if(forward>0){
      PredictStateTFT(forward, x, C, lastz, tI);
    }
    // In case of a hit, z might not be exactly the default position.
    else if(tI.m_UTLayerIdxs[3]>=0){
      PredictStateUTFUT(hits, n_ut_hits, forward, x, C, lastz, tI);
    }    
  }

  //----------------------------------------------------------------------
  // Predict UT (fixed z) <-> last UT layer.
  __device__ void PredictStateUTFUT(
    const UT::Consolidated::Hits &hits,
    int nUTHits,
    int forward,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  ) {
    Matrix5x5 F;
    if(forward>0){
      ExtrapolateUTFUTDef(lastz, x, F, tI);
    }
    else{
      // This assumes a hit in the last UT layer. Is this ok?
      ExtrapolateUTFUT(lastz, (double)hits.zAtYEq0[tI.m_UTLayerIdxs[3]], x, F, tI);
    }
    C = similarity_5_5(F, C);
  }

  //----------------------------------------------------------------------
  // Predict UT (fixed z) <-> last UT layer.
  __device__ void PredictStateUTFUT(
    int forward,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  ) {
    // TODO: Only valid for the forward direction for now...maybe not safe?
    Matrix5x5 F;
    ExtrapolateUTFUTDef(lastz, x, F, tI);
    C = similarity_5_5(F, C);
  }

  //----------------------------------------------------------------------
  // Predict T <-> T.
  __device__ void PredictStateT(
    const SciFi::Consolidated::Hits &hits,
    uint layer,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  ) {    
    double z;
    Matrix5x5 F;
    SymMatrix5x5 Q;    

    // Check if this layer has a hit.
    if(tI.m_SciFiLayerIdxs[layer]>=0){
      const uint32_t idx = (uint32_t)tI.m_SciFiLayerIdxs[layer];
      double z0 = (double)hits.z0[idx];
      double y0 = (double)hits.yMin(idx);
      double dydz = 1./hits.dzdy(idx);
      z = (lastz*x[3] - z0*dydz - x[1] + y0)/(x[3] - dydz);
      double DzDy = -1./(x[3] - dydz);
      double DzDty = lastz/(x[3] - dydz)
        - (lastz*x[3] - z0*dydz - x[1] + y0)/((x[3] - dydz)*(x[3] - dydz));
      ExtrapolateInT(lastz, layer, z, DzDy, DzDty, x, F, Q, tI);
    }
    else{
      ExtrapolateInT(lastz, layer, z, x, F, Q, tI);
    }
    
    C = similarity_5_5(F, C);
    C = C + Q;
    lastz = z;
    
  }

  //----------------------------------------------------------------------
  // Predict T (fixed z) <-> first T layer.
  __device__ void PredictStateTFT(
    const SciFi::Consolidated::Hits &hits,
    int forward,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  ) {
    double z;
    Matrix5x5 F;
    SymMatrix5x5 Q;

    if(forward>0){
      if(tI.m_SciFiLayerIdxs[0]>=0){
        int idx = tI.m_SciFiLayerIdxs[0];
        double z0 = (double)hits.z0[idx];
        double y0 = (double)hits.yMin(idx);
        double dydz = 1./(double)hits.dzdy(idx);
        z = (lastz*x[3] - z0*dydz + y0)/(x[3] - dydz);
        ExtrapolateTFT(lastz, z, x, F, Q, tI);
      }
      else{
        ExtrapolateTFTDef(lastz, z, x, F, Q, tI);
      }
    }
    else{
      z = tI.m_extr->UTTExtrEndZ();
      ExtrapolateTFT(lastz, z, x, F, Q, tI);
    }

    // Transport the covariance matrix.
    C = similarity_5_5(F, C);
    C = C + Q;
    lastz = z;    
  }

  //----------------------------------------------------------------------
  // Predict T (fixed z) <-> first T layer.
  __device__ void PredictStateTFT(
    int forward,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  ) {
    double z;
    Matrix5x5 F;
    SymMatrix5x5 Q;

    if(forward>0){
      ExtrapolateTFTDef(lastz, z, x, F, Q, tI);
    }
    else{
      z = tI.m_extr->UTTExtrEndZ();
      ExtrapolateTFT(lastz, z, x, F, Q, tI);
    }

    // Transport the covariance matrix.
    C = similarity_5_5(F, C);
    C = C + Q;
    lastz = z;    
  }
  
  //----------------------------------------------------------------------
  // Update state with VELO measurement.
  __device__ void UpdateStateV(
    const Velo::Consolidated::Hits &hits,
    int forward,
    int nHit,
    Vector5 &x,
    SymMatrix5x5 &C,
    trackInfo &tI
  ) {
    // Get residual.
    //Vector2 res({hits.x[nHit]-x(0), hits.y[nHit]-x(1)});
    Vector2 res;
    res(0) = (double)hits.x[nHit]-x(0);
    res(1) = (double)hits.y[nHit]-x(1);
    
    // TODO: For now, I'm assuming xErr == yErr == 0.015 mm. This
    // roughly matches what Daniel uses in the simplified Kalman
    // filter, but needs to be checked.
    double xErr = 0.015;
    double yErr = 0.015;
    double CResTmp[3] = {xErr*xErr + C(0,0),
                         C(0,1),
                         yErr*yErr + C(1,1)};
    SymMatrix2x2 CRes(CResTmp);
    
    // Kalman formalism.
    SymMatrix2x2 CResInv = inverse(CRes);
    Vector10 K;
    multiply_S5x5_S2x2(C, CResInv, K);
    Vector5 tmpVec;
    tmpVec = K*res;
    x = x + K*res;
    SymMatrix5x5 KCrKt;
    similarity_5x2_2x2(K, CRes, KCrKt);
    
    C = C - KCrKt;

    // Update the chi2.
    double chi2Tmp = similarity_2x1_2x2(res, CResInv);
    tI.m_chi2 += chi2Tmp;
    if(forward>0) tI.m_chi2V += chi2Tmp;
  }

  //----------------------------------------------------------------------
  // Update state with a UT measurement.
  __device__ void UpdateStateUT(
    const UT::Consolidated::Hits &hits,
    uint layer,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  ) {

    // Get the hit information.
    const double slopes[4] = {0., 0.0874886, -0.0874886, 0.};
    const int nHit = tI.m_UTLayerIdxs[layer];
    const double y0 = (double)hits.yBegin[nHit];
    const double y1 = (double)hits.yEnd[nHit];
    const double x0 = (double)hits.xAtYEq0[nHit] + y0*slopes[layer];
    const double x1 = x0 + (y1-y0)*slopes[layer];
    
    // Rotate by alpha = atan(dx/dy).
    const double x2y2 = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
    Vector2 H;
    H(0) = (y1-y0)/x2y2;
    H(1) = -(x1-x0)/x2y2;

    // Get the residual.
    const double res = H[0]*x0 + H[1]*y0 - (H[0]*x[0] + H[1]*x[1]);
    double CRes;
    similarity_1x2_S5x5_2x1(H, C, CRes);
    double err2 = 1./(double)hits.weight[nHit];
    CRes += err2;
    
    // K = P*H
    Vector5 K;
    multiply_S5x5_2x1(C, H, K);
    
    // K*S^-1
    K = K/CRes;
    x = x + res*K;
    
    // K*S*K(T)
    SymMatrix5x5 KCResKt;    
    tensorProduct(sqrt(CRes)*K, sqrt(CRes)*K, KCResKt);
    
    // P -= KSK(T)
    C = C - KCResKt;
    
    // Update chi2.
    tI.m_chi2 += res*res/CRes;
    
    // Update z.
    lastz = (double)hits.zAtYEq0[nHit];
  }
  
  //----------------------------------------------------------------------
  // Update state with a SciFi measurement.
  __device__ void UpdateStateT(
    const SciFi::Consolidated::Hits &hits,
    const int forward,
    const uint layer,
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz,
    trackInfo &tI
  ) {
    
    // For now assume consolidated SciFi hits have the same structure
    // as SciFiHits.
    const uint32_t nHit = (uint32_t)tI.m_SciFiLayerIdxs[layer];
    const double dxdy = (double)hits.dxdy(nHit);
    const double dzdy = (double)hits.dzdy(nHit);
    const double y0 = (double)hits.yMin(nHit);
    const double y1 = (double)hits.yMax(nHit);
    const double x0 = (double)hits.x0[nHit] + y0*dxdy;
    const double x1 = x0 + (y1-y0)*dxdy;
    const double z0 = (double)hits.z0[nHit];
    const double x2y2 = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
    Vector2 H;
    H(0) = (y1-y0)/x2y2;
    H(1) = -(x1-x0)/x2y2;
        
    // Residual.
    // Take the beginning point of trajectory and rotate.
    const double res = H(0)*x0 + H(1)*y0 - (H(0)*x[0] + H(1)*x[1]);
    double CRes;
    similarity_1x2_S5x5_2x1(H, C, CRes);
    // TODO: This needs a better parametrization as a function of
    // cluster size (if we actually do clustering).
    double err2 = 1./(double)hits.w(nHit);
    CRes += err2;
    
    // K*S*K(T)
    Vector5 K;
    multiply_S5x5_2x1(C, H, K);
    //K = K/CRes;
    K = K/CRes;
    x = x + res*K;
    SymMatrix5x5 KCResKt;
    tensorProduct(sqrt(CRes)*K, sqrt(CRes)*K, KCResKt);

    // P -= KSK
    C = C - KCResKt;

    // Calculate the chi2.
    tI.m_chi2 += res*res/CRes;
    if(forward<0){
      tI.m_chi2T += res*res/CRes;
    }
    
    // Update z.
    lastz = z0 + dzdy*(x[1]-y0);

  }
  
  //----------------------------------------------------------------------
  // Extrapolate to the vertex using a straight line extrapolation.
  __device__ void ExtrapolateToVertex(
    Vector5 &x,
    SymMatrix5x5 &C,
    double &lastz
  ) {
    // Determine the z position to extrapolate to.
    // z = lastz - (x*tx + y*ty)/(tx^2 + ty^2)
    double z = lastz - (x[0]*x[2] + x[1]*x[3])/(x[2]*x[2] + x[3]*x[3]);
    Matrix5x5 F(F_diag);
    x[0] = x[0] + x[2]*(lastz - z);
    x[1] = x[1] + x[3]*(lastz - z);
    F(0,2) = lastz - z;
    F(1,3) = lastz - z;
    C = similarity_5_5(F, C);
  }
  
  //----------------------------------------------------------------------
  // Check if outliers should be removed and remove one of them.
  __device__ bool DoOutlierRemoval(trackInfo &tI){
    double maxChi2 = 0;
    int n = 0;
    for(int i=0; i<tI.m_NHits; i++){
      if(tI.m_HitChi2[i]*tI.m_HitStatus[i]>maxChi2){
        maxChi2 = tI.m_HitChi2[i];
        n=i;
      }
    }
    if(maxChi2>9){
      // If the max chi2 is greater than 9, remove the hit.
      tI.m_HitStatus[n] = 0;
      if(n<tI.m_NHitsV){
        tI.m_ndof = tI.m_ndof - 2;
        tI.m_ndofV = tI.m_ndofV - 2;
      }
      else if(n<tI.m_NHitsV + tI.m_NHitsUT){
        tI.m_ndof = tI.m_ndof - 1;
        tI.m_ndofUT = tI.m_ndofUT - 1;
      }
      else{
        tI.m_ndof -= 1;
        tI.m_ndofT -= 1;
      }
      return true;
    }
    return false;
  }
  
//----------------------------------------------------------------------
// All of the extrapolations go here.

  //----------------------------------------------------------------------
  // Extrapolation in VELO.
  __device__ void ExtrapolateInV(
    double zFrom,
    double zTo,
    Vector5 &x,
    Matrix5x5 &F,
    SymMatrix5x5 &Q,
    trackInfo &tI
  ){    
    //cache the old state
    Vector5 x_old = x;
    //step size in z
    double dz = zTo - zFrom;
    //which set of parameters should be used
    const auto& par = tI.m_extr->Par_predictV[dz>0 ? 0 : 1];
    //uint parSet = dz>0 ? 0 : 1;
    
    //do not update if there is nothing to update
    if(dz == 0) return;

    //parametrizations for state extrapolation
    //tx
    x[2] = x_old[2] + x_old[4]*par[4]*1.0e-5*dz*((dz>0 ? zFrom : zTo) + par[5]*1.0e3);
    //x
    x[0] = x_old[0] + (x[2] + x_old[2])*0.5*dz;
    //ty 
    x[3] = x_old[3];
    //y
    x[1] = x_old[1] + x[3]*dz;
    //qop
    x[4] = x_old[4];
    
    //determine the Jacobian
    F.SetElements(F_diag);
    F(0,2) = dz;
    F(1,3) = dz;

    //tx
    F(2,4) = par[4]*(1.0e-5)*dz*( ( dz>0 ?  zFrom : zTo ) + par[5]*(1.0e3));
 
    //x
    F(0,4) = 0.5*dz*F(2,4);

    //Set noise matrix
  
    double sigt = par[1]*(1.0e-5) + par[2]*std::abs(x_old[4]);
    //sigma x/y
    double sigx = par[6]*sigt*std::abs(dz);
    //Correlation between x/y and tx/ty
    double corr = par[7];

    Q(0,0) = sigx*sigx;
    Q(1,1) = sigx*sigx;
    Q(2,2) = sigt*sigt;
    Q(3,3) = sigt*sigt;
  
    Q(0,2) = corr*sigx*sigt;
    Q(1,3) = corr*sigx*sigt;
  }

  //----------------------------------------------------------------------
  // Extrapolate in UT.
  __device__ bool ExtrapolateVUT(
    double zFrom,
    double zTo,
    Vector5 &x,
    Matrix5x5 &F,
    SymMatrix5x5 &Q,
    trackInfo &tI
  ){
  
    //cache the old state
    Vector5 x_old = x;
    //step size in z
    double dz = zTo - zFrom;
    //which set of parameters should be used
    const auto& par = tI.m_extr->Par_predictVUT[dz>0 ? 0 : 1];
    //extrapolate the current state and define noise
    if(dz>0){
      //ty
      x[3] = x_old[3]
        + par[0]*std::copysign(1.0, x[1] )*x_old[4]*x_old[2];
                 
      double tyErr = par[3]*std::fabs(x_old[4]);

      //y 
      x[1] = x_old[1] 
        + (par[5]*x_old[3] + (1-par[5])*x[3])*dz;
      
      double yErr = par[6]*std::abs(dz*x_old[4]);
      
      //tx  
      double coeff = par[8] *1e1
        + par[9] *1e-2*zFrom
        + par[10]*1e2 *x_old[3]*x_old[3];
      
      double a = x_old[2]/std::sqrt(1.0 + x_old[2]*x_old[2] + x_old[3]*x_old[3])
        - x_old[4]*coeff;
      
      //Check that the track is not deflected
      if(std::fabs(a)>=1) return false;
      
      x[2] = a*sqrt(1.0/(1.0 - a*a)*(1.0 + x[3]*x[3]));
      
      double txErr = par[15]*std::fabs(x_old[4]);

      //x
      double zmag = par[16]*1e3
                  + par[17]     *zFrom
                  + par[18]*1e-5*zFrom*zFrom
                  + par[19]*1e3 *x_old[3]*x_old[3];
      
      x[0] = x_old[0] + (zmag - zFrom)*x_old[2] + (zTo-zmag)*x[2];
      
      double xErr = par[20]*std::abs(dz*x_old[4]);
    
      //calculate jacobian 
      //ty
      F(3,0) = 0;
      F(3,1) = 0;
      F(3,2) = par[0]*x_old[4];
      F(3,3) = 1;
      F(3,4) = par[0]*x_old[2];
      //y
      double DyDty = (1 - par[5])*dz;
      F(1,0) = 0.0;
      F(1,1) = 1.0;
      F(1,2) = DyDty*F(3,2);  
      F(1,3) = dz;
      F(1,4) = DyDty*F(3,4);
    
      //tx
      double sqrtTmp = std::sqrt((1.0 - a*a)*(1.0 + x[3]*x[3]));
      double DtxDty = a*x[3]*1.0/sqrtTmp;
      double DtxDa = sqrtTmp/((a*a - 1)*(a*a - 1));
      F(2,0) = 0; 
      F(2,1) = 0;
      
      sqrtTmp = std::sqrt(1 + x_old[2]*x_old[2] + x_old[3]*x_old[3]);
      F(2,2) = DtxDa*(1 + x_old[3]*x_old[3])/(sqrtTmp*(1 + x_old[2]*x_old[2] + x_old[3]*x_old[3])) + DtxDty*F(3,2);
      
      F(2,3) = DtxDa*(-x_old[2]*x_old[3]/
                      (sqrtTmp*(1 + x_old[2]*x_old[2] + x_old[3]*x_old[3]))
                      - x_old[4]*2*par[10]*1e2*x_old[3])
        + DtxDty*F(3,3);
      
      F(2,4) = DtxDa*(-coeff) + DtxDty*F(3,4);
      
      //x
      F(0,0) = 1;
      F(0,1) = 0;
      F(0,2) = (zmag - zFrom) + (zTo - zmag)*F(2,2);
      
      F(0,3) = (zTo - zmag)*F(2,3) + (x_old[2] - x[2])*2*par[19]*1e3*x_old[3];
      
      F(0,4) = (zTo - zmag)*F(2,4);
      
      //qop
      F(4,0) = 0;
      F(4,1) = 0;
      F(4,2) = 0;
      F(4,3) = 0;
      F(4,4) = 1;
      
      //add noise
      Q(0,0) = xErr*xErr;
      Q(0,2) = par[4]*xErr*txErr;
      Q(1,1) = yErr*yErr;
      Q(1,3) = par[21]*yErr*tyErr;
      Q(2,2) = txErr*txErr;
      Q(3,3) = tyErr*tyErr;

    }
    else{
      //ty
      x[3] = x_old[3]
        + par[0]*std::copysign(1.0, x[1] )*x_old[4]*x_old[2];
      
      double tyErr = par[3]*std::fabs(x_old[4]);
      
      //y 
      x[1] = x_old[1] 
        + (par[5]*x_old[3] + (1 - par[5])*x[3])*dz;
      
      double yErr = par[6]*std::abs(dz*x_old[4]) ;
      
      //tx  
      double coeff = par[8] *1e1
        + par[9] *1e-2*zTo
        + par[10]*1e2 *x_old[3]*x_old[3];
      
      double a = x_old[2]/std::sqrt(1.0 + x_old[2]*x_old[2] + x_old[3]*x_old[3])
      - x_old[4]*coeff;
      
      //Check that the track is not deflected
      if(std::fabs(a)>=1) return false;
      
      x[2] = a*sqrt(1.0/(1.0 - a*a)*(1.0 + x[3]*x[3]));
      double txErr = par[15]*std::fabs(x_old[4]);
      
      //x
      double zmag = par[16]*1e3
        + par[17]     *zTo
        + par[18]*1e-5*zTo*zTo
        + par[19]*1e3 *x_old[3]*x_old[3];
      
      x[0] = x_old[0] + (zmag - zFrom)*x_old[2] + (zTo - zmag)*x[2];
      
      double xErr = par[20]*std::abs(dz*x_old[4]);
      
      //calculate jacobian 
      //ty
      F(3,0) = 0;
      F(3,1) = 0;
      F(3,2) = par[0]*x_old[4];
      F(3,3) = 1;
      F(3,4) = par[0]*x_old[2];
      //y
      double DyDty = (1 - par[5])*dz;
      F(1,0) = 0.0;
      F(1,1) = 1.0;
      F(1,2) = DyDty*F(3,2);  
      F(1,3) = dz;
      F(1,4) = DyDty*F(3,4);
      
      //tx
      double sqrtTmp = std::sqrt((1 - a*a)*(1 + x[3]*x[3]));
      double DtxDty = a*x[3]*1.0/sqrtTmp;
      double DtxDa = sqrtTmp/((a*a - 1)*(a*a - 1));
      F(2,0) = 0; 
      F(2,1) = 0;
      
      sqrtTmp = std::sqrt(1 + x_old[2]*x_old[2] + x_old[3]*x_old[3]);
      F(2,2) = DtxDa*(1 + x_old[3]*x_old[3])/
        (sqrtTmp*(1 + x_old[2]*x_old[2] + x_old[3]*x_old[3]))
        + DtxDty*F(3,2);
      
      F(2,3) = DtxDa*(-x_old[2]*x_old[3]/
                      (sqrtTmp*(1 + x_old[2]*x_old[2] + x_old[3]*x_old[3]))
                      - x_old[4]*2*par[10]*1e2*x_old[3])
        + DtxDty*F(3,3);
      
      F(2,4) = DtxDa*(-coeff)
        + DtxDty*F(3,4);
      
      //x
      F(0,0) = 1;
      F(0,1) = 0;
      F(0,2) = (zmag - zFrom) + (zTo - zmag)*F(2,2);
      
      F(0,3) = (zTo - zmag)*F(2,3) + (x_old[2] - x[2])*2*par[19]*1e3*x_old[3];
      
      F(0,4) = (zTo - zmag)*F(2,4);
      
      //qop
      F(4,0) = 0;
      F(4,1) = 0;
      F(4,2) = 0;
      F(4,3) = 0;
      F(4,4) = 1;
      
      //add noise
      Q(0,0) = xErr*xErr;
      Q(0,2) = par[4]*xErr*txErr;
      Q(1,1) = yErr*yErr;
      Q(1,3) = par[21]*yErr*tyErr;
      Q(2,2) = txErr*txErr;
      Q(3,3) = tyErr*tyErr;
    }

    return true;
    
  }

  //----------------------------------------------------------------------
  // Get noise in VELO-UT, backwards.
  __device__ void GetNoiseVUTBackw(
    double zFrom,
    double zTo,
    Vector5 &x,
    SymMatrix5x5 &Q,
    trackInfo &tI
  ){
  
    //step size in z
    double dz = zTo - zFrom;
    //which set of parameters should be used
    const auto& par = tI.m_extr->Par_predictVUT[1];

    //ty
    double tyErr = par[3]*std::fabs(x[4]);

    //y 
    double yErr = par[6]*std::abs(dz*x[4]) ;

    //tx  
    double txErr = par[15]*std::fabs(x[4]);

    //x
    double xErr = par[20]*std::abs(dz*x[4]);
  
    //add noise
    Q(0,0) = xErr*xErr;
    Q(0,2) = par[4]*xErr*txErr;
    Q(1,1) = yErr*yErr;
    Q(1,3) = par[21]*yErr*tyErr;
    Q(2,2) = txErr*txErr;
    Q(3,3) = tyErr*tyErr;

  }

  //----------------------------------------------------------------------
  // Extrapolate in UT.
  __device__ void ExtrapolateInUT(
    double zFrom,
    uint nLayer,
    double zTo,
    Vector5 &x,
    Matrix5x5 &F,
    SymMatrix5x5 &Q,
    trackInfo &tI
  ){
  
    //In case no specific z-position is set the default z position is used
    if(zFrom==zTo) zTo=tI.m_extr->Par_UTLayer[0][nLayer];
    //cache the old state
    Vector5 x_old = x;
    //step size in z
    double dz = zTo - zFrom;
    //which set of parameters should be used
    const auto& par = tI.m_extr->Par_predictUT[( dz > 0 ? nLayer - 1 : (5 - nLayer) )];

    //extrapolate state vector
    //tx 
    x[2] += dz*(par[5] *1.e-1*x[4]
                + par[6] *1.e3*x[4]*x[4]*x[4]
                + par[7] *1e-7*x[1]*x[1]*x[4]
                );
    //x
    x[0] += dz*(par[0]*x_old[2] + (1 - par[0])*x[2]);
    //ty
    x[3] += par[10]*x[4]*x[2]*std::copysign(1.0, x[1] ); 
    //y
    x[1] += dz*(par[3]*x_old[3] + (1 - par[3])*x[3]);

    F(2,0) = 0;
    F(2,1) = 2*dz*par[7]*1e-7*x_old[1]*x[4];
    F(2,2) = 1;
    F(2,3) = 0;
    F(2,4) = dz*(par[5]*1.e-1
                 + 3*par[6]*1.e3*x[4]*x[4]
                 + par[7]*1e-7*x_old[1]*x_old[1]);
  
    F(0,0) = 1; 
    F(0,1) = dz*(1 - par[0])*F(2,1);
    F(0,2) = dz;
    F(0,3) = 0;
    F(0,4) = dz*(1 - par[0])*F(2,4);
 
    F(3,0) = 0;
    F(3,1) = 0;
    F(3,2) = par[10]*x[4]*std::copysign(1.0, x[1] );
    F(3,3) = 1;
    F(3,4) = par[10]*x[2]*std::copysign(1.0, x[1] );
  
    F(1,0) = 0; 
    F(1,1) = 1;
    F(1,2) = dz*(1 - par[3])*F(3,2);
    F(1,3) = dz;
    F(1,4) = dz*(1 - par[3])*F(3,4);
 
    F(4,0) = 0;
    F(4,1) = 0;
    F(4,2) = 0;
    F(4,3) = 0;
    F(4,4) = 1;

    //Define noise
    double xErr  = par[2]*std::fabs(dz*x_old[4]);  
    double yErr  = par[4]*std::fabs(dz*x_old[4]);  
    double txErr = par[12]*std::fabs(x_old[4]);  
    double tyErr = par[15]*std::fabs(x_old[4]);  
    
    //Add noise
    Q(0,0) = xErr*xErr;
    Q(0,2) = par[14]*xErr*txErr;
    Q(1,1) = yErr*yErr;
    Q(1,3) = par[17]*yErr*tyErr;
    Q(2,2) = txErr*txErr;
    Q(3,3) = tyErr*tyErr;
  }

  //----------------------------------------------------------------------
  // Extrapolate UTFUT(?)
  __device__ void ExtrapolateUTFUTDef(
    double &zFrom,
    Vector5 &x,
    Matrix5x5 &F,
    trackInfo &tI
  ){
    //Use the start position of the UTTF extrapolation as default z value
    ExtrapolateUTFUT(zFrom, tI.m_extr->Par_UTLayer[0][3], x, F, tI);
    zFrom = tI.m_extr->Par_UTLayer[0][3];
  }

  //----------------------------------------------------------------------
  // Extrapolate UTFUT(?)
  __device__ void ExtrapolateUTFUT(
    double zFrom,
    double zTo,
    Vector5 &x,
    Matrix5x5 &F,
    trackInfo &tI
  ){
    //cache the old state
    Vector5 x_old = x;
    //step size in z
    double dz = zTo - zFrom;
    //which parameters should be used?
    const auto& par = tI.m_extr->Par_predictUTFUT[0]; 
 
    //do the extrapolation of the state vector
    //tx
    x[2] = x_old[2] + par[0]*x_old[4]*dz;
    //x
    x[0] = x_old[0] + (x[2] + x_old[2])*0.5*dz;
    //y
    x[1] = x_old[1] + x_old[3]*dz;

    //Jacobian
    F.SetElements(F_diag);

    //tx
    F(2,4) = par[0]*dz;
    //x
    F(0,2) = dz;  
    F(0,4) = 0.5*dz*F(2,4);
    //y
    F(1,3) = dz; 
  }

  //----------------------------------------------------------------------
  // Extrapolate UT-T
  __device__ void ExtrapolateUTT(
    Vector5 &x,
    Matrix5x5 &F,
    SymMatrix5x5 &Q,
    trackInfo &tI
  ){
    
    //cache old state
    Vector5 x_old = x;

    const auto& par = tI.m_extr->Par_predictUTTF[0];

    //extrapolating from last UT layer (z=2642.5) to fixed z in T (z=7855)

    //determine the momentum at this state from the momentum saved in the state vector
    //(representing always the PV qop) 
    double qopHere = x[4] + x[4]*1e-4*par[18] + x[4]*std::abs(x[4])*par[19]; //TODO make this a tuneable parameter 

    //do the actual extrapolation
    double der_tx[4], der_ty[4], der_qop[4];//, der_x[4], der_y[4];
    extrapUTT(tI.m_extr->UTTExtrBeginZ(), tI.m_extr->UTTExtrEndZ(), QUADRATICINTERPOLATION, x[0], x[1], x[2], x[3], qopHere, der_tx, der_ty, der_qop, tI);

    //apply additional correction
    x[0] += par[9] *x_old[4]*1e2
          + par[10]*x_old[4]*x_old[4]*1e5
          + par[11]*x_old[4]*x_old[4]*x_old[4]*1e10;
    x[1] += par[3] *x_old[4]*1e2;
    x[2] += par[6] *x_old[4]
          + par[7] *x_old[4]*x_old[4]*1e5
          + par[8] *x_old[4]*x_old[4]*x_old[4]*1e8;
    x[3] += par[0] *x_old[4];
           
    //Set jacobian matrix 
    //TODO study impact of der_x, der_y 
    //ty
    F(3,0) = 0;//der_x[3];
    F(3,1) = 0;//der_y[3];
    F(3,2) = der_tx[3];
    F(3,3) = der_ty[3];
    F(3,4) = der_qop[3]*(1 + 2*std::abs(x[4])*par[18])
      + par[0]
      + 2*par[1]*x_old[4]*1e5
      + 3*par[2]*x_old[4]*x_old[4]*1e8;
    //y
    F(1,0) = 0;//der_x[1];
    F(1,1) = 1;//der_y[1]; 
    F(1,2) = der_tx[1];  
    F(1,3) = der_ty[1];
    F(1,4) = der_qop[1]*(1 + 2*std::abs(x[4])*par[18])
      + par[3]*1e2
      + 2*par[4]*x_old[4]*1e5
      + 3*par[5]*x_old[4]*x_old[4]*1e8;
  
    //tx
    F(2,0) = 0;//der_x[2]; 
    F(2,1) = 0;//der_y[2];
    F(2,2) = der_tx[2];
    F(2,3) = der_ty[2];
    F(2,4) = der_qop[2]*(1 + 2*std::abs(x[4])*par[18])
      + par[6]
      + 2*par[7]*x_old[4]*1e5
      + 3*par[8]*x_old[4]*x_old[4]*1e8;
  
    //x
    F(0,0) = 1;//der_x[0];
    F(0,1) = 0;//der_y[0];
    F(0,2) = der_tx[0];
    F(0,3) = der_ty[0];
    F(0,4) = der_qop[0]*(1 + 2*std::abs(x[4])*par[18])
      + par[9]*1e2
      + 2*par[10]*x_old[4]*1e5
      + 3*par[11]*x_old[4]*x_old[4]*1e10;
   
    //qop
    F(4,0) = 0;
    F(4,1) = 0;
    F(4,2) = 0;
    F(4,3) = 0;
    F(4,4) = 1;

    //Define noise
    double xErr  = par[13]*1e2*std::abs(x_old[4]); 
    double yErr  = par[16]*1e2*std::abs(x_old[4]); 
    double txErr = par[12]*std::abs(x_old[4]); 
    double tyErr = par[15]*std::abs(x_old[4]); 

    //Add noise
    Q(0,0) = xErr*xErr;
    Q(0,2) = par[14]*xErr*txErr;
    Q(1,1) = yErr*yErr;
    Q(1,3) = par[17]*yErr*tyErr;
    Q(2,2) = txErr*txErr;
    Q(3,3) = tyErr*tyErr;
  }

  //----------------------------------------------------------------------
  // Get noise for UT-T (backwards).
  __device__ void GetNoiseUTTBackw(
    const Vector5 &x,
    SymMatrix5x5 &Q,
    trackInfo &tI
  ){
    const auto& par = tI.m_extr->Par_predictUTTF[1];
 
    //Define noise
    double xErr  = par[13]*1e2*std::abs(x[4]); 
    double yErr  = par[16]*1e2*std::abs(x[4]); 
    double txErr = par[12]*std::abs(x[4]); 
    double tyErr = par[15]*std::abs(x[4]); 

    //Add noise
    Q(0,0) = xErr*xErr;
    Q(0,2) = par[14]*xErr*txErr;
    Q(1,1) = yErr*yErr;
    Q(1,3) = par[17]*yErr*tyErr;
    Q(2,2) = txErr*txErr;
    Q(3,3) = tyErr*tyErr;
  }

  //----------------------------------------------------------------------
  // Extrapolate in the SciFi.
  __device__ void ExtrapolateInT(
    double zFrom,
    uint nLayer,
    double &zTo,
    Vector5 &x,
    Matrix5x5 &F,
    SymMatrix5x5 &Q,
    trackInfo &tI
  ){
    //determine next z position:
    //use the straigt line extrapolation in y
    //and calculate the intersection with the detector layer
    double z0 = tI.m_extr->Par_TLayer[0][nLayer]; 
    double y0 = 0;
    double dydz = tI.m_extr->Par_TLayer[1][nLayer]; 
    zTo=(zFrom*x[3]-z0*dydz-x[1]+y0)/(x[3]-dydz);
    //TODO use this derivatives: Tested: it does not help. Remove it at some point!
    double DzDy  = -1.0/(x[3]-dydz);
    double DzDty = zFrom/(x[3]-dydz)-(zFrom*x[3]-z0*dydz-x[1]+y0)/((x[3]-dydz)*(x[3]-dydz));
  
    ExtrapolateInT(zFrom, nLayer, zTo, DzDy, DzDty, x, F, Q, tI);
  }

  //----------------------------------------------------------------------
  // Extrapolate in the SciFi.
  __device__ void ExtrapolateInT(
    double zFrom,
    uint nLayer,
    double zTo,
    double DzDy,
    double DzDty,
    Vector5 &x,
    Matrix5x5 &F,
    SymMatrix5x5 &Q,
    trackInfo &tI
  ){
    
    //cache the old state
    Vector5 x_old = x;
    //step size in z
    double dz = zTo - zFrom;
    //which set of parameters should be used
    //Reminder: backward T station label is different for the iPar definition 
    //44 42 40 38     36 34 32 30    28 26 24 
    //|  |  |  |      |  |  |  |     |  |  |  | 
    //|  |  |  |      |  |  |  |     |  |  |  | 
    //45 43 41 39     37 35 33 31    29 27 25 
    int iPar = ( dz > 0 ? 2*nLayer - 2 : (42 - 2*nLayer) );
    if(x[1] < 0) iPar += 1;
    const auto& par = tI.m_extr->Par_predictT[iPar];

    //predict state
    //tx 
    x[2] += dz*(par[5]*1.e-1*x[4]
                + par[6]*1.e3 *x[4]*x[4]*x[4]
                + par[7]*1e-7 *x[1]*x[1]*x[4]);
    //x
    x[0] += dz*(par[0]*x_old[2] + (1 - par[0])*x[2]);
    //ty
    x[3] += par[10]*x[4]*x[4]*x[1]; 
    //y
    x[1] += dz*(par[3]*x_old[3] + (1 - par[3])*x[3]);

    //calculate jacobian

    double dtxddz = par[5]*1.e-1*x[4]
      + par[6]*1.e3 *x[4]*x[4]*x[4]
      + par[7]*1e-7 *x[1]*x[1]*x[4];
             
    F(2,0) = 0;
    F(2,1) = 2*dz*par[7]*1e-7*x_old[1]*x[4]
      + dtxddz*DzDy;
    F(2,2) = 1;
    F(2,3) = dtxddz*DzDty;
    F(2,4) = dz*(par[5]*1.e-1
                 + 3*par[6]*1.e3*x[4]*x[4]
                 + par[7]*1e-7*x_old[1]*x_old[1]);
  
    double dxddz = par[0]*x_old[2] + (1 - par[0])*x[2];
    F(0,0) = 1; 
    F(0,1) = dz*(1 - par[0])*F(2,1) + dxddz*DzDy;
    F(0,2) = dz;
    F(0,3) = dz*(1-par[0])*F(2,3) + dxddz*DzDty;
    F(0,4) = dz*(1-par[0])*F(2,4);
 
 
    F(3,0) = 0;
    F(3,1) = 0;
    F(3,2) = 0;
    F(3,3) = 1;
    F(3,4) = 2*par[10]*x[4];
  
    F(1,0) = 0; 
    F(1,1) = 1;
    F(1,2) = 0;
    F(1,3) = dz;
    F(1,4) = dz*(1 - par[3])*F(3,4);
  
 
    F(4,0) = 0;
    F(4,1) = 0;
    F(4,2) = 0;
    F(4,3) = 0;
    F(4,4) = 1;

    //Define noise
    double xErr  = par[2] *std::fabs(dz*x_old[4]);  
    double yErr  = par[4] *std::fabs(dz*x_old[4]);  
    double txErr = par[12]*std::fabs(x_old[4]);  
    double tyErr = par[15]*std::fabs(x_old[4]);  
  
    Q(0,0) = xErr*xErr;
    Q(0,2) = par[14]*xErr*txErr;
    Q(1,1) = yErr*yErr;
    Q(1,3) = par[17]*yErr*tyErr;
    Q(2,2) = txErr*txErr;
    Q(3,3) = tyErr*tyErr;
    
  }

  //----------------------------------------------------------------------
  // Extrapolate T(fixed z) <-> first T layer (no hit)
  __device__ void ExtrapolateTFTDef(
    double zFrom,
    double &zTo,
    Vector5 &x,
    Matrix5x5 &F,
    SymMatrix5x5 &Q,
    trackInfo &tI
  ){
    double z0 = tI.m_extr->UTTExtrEndZ();
    double y0 = 0;
    double dydz = tI.m_extr->Par_TLayer[1][0];
    zTo = (zFrom*x[3] - z0*dydz - x[1] + y0)/(x[3] - dydz);
    ExtrapolateTFT(zFrom, zTo, x, F, Q, tI);
  }
  
  //----------------------------------------------------------------------
  // Extrapolate TFT.
  __device__ void ExtrapolateTFT(
    double zFrom,
    double &zTo,
    Vector5 &x,
    Matrix5x5 &F,
    SymMatrix5x5 &Q,
    trackInfo &tI
  ){
    
    //cache the old state
    Vector5 x_old = x;
    //step size in z
    double dz = zTo - zFrom;
    //which parameters should be used?
    const auto& par = tI.m_extr->Par_predictTFT[ dz>0 ? 0 : 1 ]; 
 
    //do the extrapolation of the state vector
    //tx
    x[2] = x_old[2]
      + par[5]*x_old[4]*dz
      + 1e4*par[6]*x_old[4]*dz*x_old[4]*dz*x_old[4]*dz;
    //x
    x[0] = x_old[0]
      + ((1 - par[8])*x[2] + par[8]*x_old[2])*dz;
    //ty 
    x[3] = x_old[3]
      + par[0]*(x_old[4]*dz)*(x_old[4]*dz);
    //y
    x[1] = x_old[1]
      + (x[3] + x_old[3])*0.5*dz;
    //qop
    x[4] = x_old[4];

    //Jacobian
    F.SetElements(F_diag);
    F(0,2) = dz;  
    F(1,3) = dz;  

    //tx
    F(2,4) = par[5]*dz
      + 3*1e4*par[6]*dz*dz*dz*x_old[4]*x_old[4];
    //x
    F(0,4) = (1 - par[8])*dz*F(2,4);
    //ty
    F(3,4) = 2*par[0]*x_old[4]*dz*dz;
    //y
    F(1,4) = 0.5*dz*F(3,4);

    //Set noise: none
    //Should be already initialized to 0
    Q(0,0) = 0; 

  }

  __device__ int extrapUTT(
    double zi,double zf, int quad_interp,
    double& x,double& y,double& tx,double& ty, double qOp,
    double *der_tx,double *der_ty,double *der_qop, trackInfo &tI)
  // extrapolation from plane zi to plane zf, from initial state (x,y,tx,ty,qop) to final state (x,y,tx,ty)
  // the bending from origin to zi is approxmated by adding bend*qop to x/zi.
  // quad_inperp (logical): if true, the quadratic interpolation is used (better, with a little bit more computations)
  // XGridOption and YGridOption describe the choice of xy grid. By default, it is 1 (equally spaced values)   
  {
    double qop = tI.m_extr->m_qop_flip ? -qOp : qOp; 
 
    double xx(0),yy(0),dx,dy,ux,uy;
    int ix,iy;
    //if(fabs(x)>Xmax||fabs(y)>Ymax) return 0;
    switch(tI.m_extr->XGridOption) {
    case 1: xx = x/tI.m_extr->Xmax; break;
    case 2: xx = (x/tI.m_extr->Xmax)*(x/tI.m_extr->Xmax); if(x<0) xx = -xx; break;
    case 3: xx = x/tI.m_extr->Xmax; xx = xx*xx*xx; break;
    case 4: xx = asin(x/tI.m_extr->Xmax)*2/M_PI; break;
    }
    switch(tI.m_extr->YGridOption) {
    case 1: yy = y/tI.m_extr->Ymax; break;
    case 2: yy = (y/tI.m_extr->Ymax)*(y/tI.m_extr->Ymax); if(y<0) yy = -yy; break;
    case 3: yy = y/tI.m_extr->Ymax; yy = yy*yy*yy; break;
    case 4: yy = asin(y/tI.m_extr->Ymax)*2/M_PI; break;
    }
    dx = tI.m_extr->Nbinx*(xx+1)/2; ix = dx; dx -= ix;
    dy = tI.m_extr->Nbiny*(yy+1)/2; iy = dy; dy -= iy;

    double bendx = tI.m_extr->BENDX+tI.m_extr->BENDX_X2*(x/zi)*(x/zi)+tI.m_extr->BENDX_Y2*(y/zi)*(y/zi);
    double bendy = tI.m_extr->BENDY_XY*(x/zi)*(y/zi);
    ux = (tx-x/zi-bendx*qop)/tI.m_extr->Dtxy; uy = (ty-y/zi-bendy*qop)/tI.m_extr->Dtxy;
    //if(fabs(ux)>2||fabs(uy)>2) return 0;

    StandardCoefs c;

    if(quad_interp) {
      double gx,gy;
      gx = dx-.5; gy = dy-.5;
      //if(gx*gx+gy*gy>.01) return 0;
      if(ix<=0) { ix = 1; gx -= 1.; }
      if(ix>=tI.m_extr->Nbinx-1) { ix = tI.m_extr->Nbinx-2; gx += 1.; }
      if(iy<=0) { iy = 1; gy -= 1.; }
      if(iy>=tI.m_extr->Nbiny-1) { iy = tI.m_extr->Nbiny-2; gy += 1.; }

      int rx,ry,sx,sy;
      rx = (gx>=0); sx = 2*rx-1; ry = (gy>=0); sy = 2*ry-1;
      StandardCoefs c00,cp0,c0p,cn0,c0n,cadd;
      c00 = tI.m_extr->C[ix][iy];
      cp0 = tI.m_extr->C[ix+1][iy];
      c0p = tI.m_extr->C[ix][iy+1];
      c0n = tI.m_extr->C[ix][iy-1];
      cn0 = tI.m_extr->C[ix-1][iy];
      cadd = tI.m_extr->C[ix+sx][iy+sy];
      double gxy = gx*gy, gx2 = gx*gx, gy2 = gy*gy, g2 = gx*gx+gy*gy;
            
      c = c00*(1-g2) + (cp0*(gx2+gx) + cn0*(gx2-gx) + c0p*(gy2+gy) + c0n*(gy2-gy))*.5
        + ((c00+cadd)*sx*sy - cp0*rx*sy + cn0*(!rx)*sy - c0p*ry*sx + c0n*(!ry)*sx)*gxy;
    }
    else {
      double ex,fx,ey,fy;
      int jx,jy;
      if(dx<.5) { jx = ix-1; ex = .5+dx; fx = .5-dx; } else { jx = ix+1; ex = 1.5-dx; fx = dx-.5; }
      if(dy<.5) { jy = iy-1; ey = .5+dy; fy = .5-dy; } else { jy = iy+1; ey = 1.5-dy; fy = dy-.5; }
      if(ix<0||ix>=tI.m_extr->Nbinx||iy<0||iy>=tI.m_extr->Nbiny ||
         jx<0||jx>=tI.m_extr->Nbinx||jy<0||jy>=tI.m_extr->Nbiny) return 0;
      StandardCoefs c_ii = tI.m_extr->C[ix][iy];
      StandardCoefs c_ij = tI.m_extr->C[ix][jy];
      StandardCoefs c_ji = tI.m_extr->C[jx][iy];
      StandardCoefs c_jj = tI.m_extr->C[jx][jy];
      c = c_ii*ex*ey + c_ij*ex*fy + c_ji*fx*ey + c_jj*fx*fy;
    }
    x = x+tx*(zf-zi);
    y = y+ty*(zf-zi);
    tx = tx;
    ty = ty;
    
    for(int k=0; k<4; k++) der_tx[k] = der_ty[k] = der_qop[k] = 0;
    // corrections to straight line -------------------------
    double fq = qop*tI.m_extr->PMIN;
    // x and tx ---------
    double ff = 1;
    double term1,term2;
    for(int deg=0; deg<c.degx1; deg++) {
      term1 = c.x00(deg)+c.x10(deg)*ux+c.x01(deg)*uy;
      term2 = c.tx00(deg)+c.tx10(deg)*ux+c.tx01(deg)*uy;
      der_qop[0] += (deg+1)*term1*ff;
      der_qop[2] += (deg+1)*term2*ff;
      ff *= fq;
      x  += term1*ff;
      tx += term2*ff;
      der_tx[0] += c.x10(deg)*ff;
      der_ty[0] += c.x01(deg)*ff;
      der_tx[2] += c.tx10(deg)*ff;
      der_ty[2] += c.tx01(deg)*ff;
    }
    
    for(int deg=c.degx1; deg<c.degx2; deg++) {
      der_qop[0] += (deg+1)*c.x00(deg)*ff;
      der_qop[2] += (deg+1)*c.tx00(deg)*ff;
      ff *= fq;
      x  += c.x00(deg)*ff;
      tx += c.tx00(deg)*ff;
    }
    
    // y and ty ---------
    ff = 1;
    for(int deg=0; deg<c.degy1; deg++) {
      term1 = c.y00(deg)+c.y10(deg)*ux+c.y01(deg)*uy;
      term2 = c.ty00(deg)+c.ty10(deg)*ux+c.ty01(deg)*uy;
      der_qop[1] += (deg+1)*term1*ff;
      der_qop[3] += (deg+1)*term2*ff;
      ff *= fq;
      y  += term1*ff;
      ty += term2*ff;
      der_tx[1] += c.y10(deg)*ff;
      der_ty[1] += c.y01(deg)*ff;
      der_tx[3] += c.ty10(deg)*ff;
      der_ty[3] += c.ty01(deg)*ff;
    }

    for(int deg=c.degy1; deg<c.degy2; deg++) {
      der_qop[1] += (deg+1)*c.y00(deg)*ff;
      der_qop[3] += (deg+1)*c.ty00(deg)*ff;
      ff *= fq;
      y  += c.y00(deg)*ff;
      ty += c.ty00(deg)*ff;
    }

    for(int k=0; k<4; k++) {
      der_qop[k] *= tI.m_extr->PMIN;
      der_tx[k] /= tI.m_extr->Dtxy;
      der_ty[k] /= tI.m_extr->Dtxy;
    }
    // from straight line
    der_tx[0] += zf-zi; der_ty[1] += zf-zi;
    der_tx[2] += 1; der_ty[3] += 1;

    if(tI.m_extr->m_qop_flip){
      der_qop[0]=-der_qop[0];
      der_qop[1]=-der_qop[1];
      der_qop[2]=-der_qop[2];
      der_qop[3]=-der_qop[3];
    }

    return 1;
  }
