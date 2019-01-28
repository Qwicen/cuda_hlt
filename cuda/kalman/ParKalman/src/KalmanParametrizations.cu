#include <iostream>
#include <fstream>
#include <string.h>
#include "KalmanParametrizations.cuh"

namespace ParKalmanFilter {
  
  ////////////////////////////////////////////////////////////////////////
  // Read parameters.
  ////////////////////////////////////////////////////////////////////////

  //----------------------------------------------------------------------
  //  read parameters from file - here for the extrapolation UT -> T
  __host__ void KalmanParametrizations::read_params_UTT(std::string file){
    std::string line;
    std::ifstream myfile(file);
    if(!myfile.is_open())
      throw StrException("Failed to open parameter file.");
    myfile >> ZINI >> ZFIN >> PMIN
           >> BENDX >> BENDX_X2 >> BENDX_Y2 >> BENDY_XY
           >> Txmax >> Tymax >> XFmax >> Dtxy;
    myfile >> Nbinx >> Nbiny >> XGridOption >> YGridOption
           >> DEGX1 >> DEGX2 >> DEGY1 >> DEGY2;  
    for(int ix=0; ix<Nbinx; ix++)
      for(int iy=0; iy<Nbiny; iy++)
        C[ix][iy].Read(myfile);
    Xmax = ZINI*Txmax;
    Ymax = ZINI*Tymax;
  }
  
  __device__ __host__ double KalmanParametrizations::UTTExtrEndZ() const{
    return ZFIN;
  }

  __device__ __host__ double KalmanParametrizations::UTTExtrBeginZ() const{
    return ZINI;
  }

  __device__ __host__ double KalmanParametrizations::VUTExtrEndZ() const{
    return Par_UTLayer[0][0];
  }
  
}


