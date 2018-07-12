//#include "definitions.h"
#include "global.h"
#include "../../cuda/velo/common/include/VeloDefinitions.cuh"




  class AdaptivePVTrack 
  {
  public:
    AdaptivePVTrack( VeloState& track, XYZPoint& vtx) ;
    void updateCache( const XYZPoint& vtx ) ;
    double weight() const { return m_weight ; }
    void setWeight(double w) { m_weight = w ;}
    const double *  halfD2Chi2DX2() const { return m_halfD2Chi2DX2 ; }
    const XYZPoint&  halfDChi2DX() const { return m_halfDChi2DX ; }
    double chi2() const { return m_chi2 ; }
    double chi2( const XYZPoint& vtx ) const ;
    VeloState track() const { return m_track ; }
  private:
    double m_weight ;
     VeloState m_track ;
    VeloState m_state ;
    //express symmetrical amtrices as arrays in in packed representation element m(i,j) (j <= i) is supposed to be in array element  (i * (i + 1)) / 2 + j


    double m_invcov[3] ;
    double m_halfD2Chi2DX2[6] ;
    XYZPoint m_halfDChi2DX{1.,2.,3.}  ;
    double m_chi2 ;
    double m_H[6];
  } ;