// $Id: PrUTMagnetTool.h,v 1.5 2009-10-30 13:20:50 wouter Exp $ 
#ifndef PRUTMAGNETTOOL_H
#define PRUTMAGNETTOOL_H 1






#include "PrTableForFunction.h"
#include "MagneticFieldGrid.h"
#include "MagneticFieldGridReader.h"

#include <algorithm>
#include <iostream>
#include <math.h>  
#include <array>

#include "TROOT.h"
#include "TMath.h"
#include "TH1D.h"
#include "Math/Vector3D.h"
#include "Math/Point3D.h"
#include "Math/SMatrix.h"
using namespace std;


    typedef ROOT::Math::XYZVector XYZVector;
    typedef ROOT::Math::SMatrix<double, 3, 3> Matrix3x3;
    typedef ROOT::Math::XYZPoint XYZPoint;
    typedef XYZVector FieldVector ;
    typedef Matrix3x3 FieldGradient ;




  /** @class PrUTMagnetTool PrUTMagnetTool.h newtool/PrUTMagnetTool.h
   *
   *  Magnet tool for Pr
   *
   *  @author Mariusz Witek
   *  @date   2006-09-25
   *  @update for A-Team framework 2007-08-20 SHM
   *
   */
class DeSTDetector;
class DeSTLayer;
    class PrTableForFunction;

class PrUTMagnetTool  {
public:


  
  /// Standard constructor
  PrUTMagnetTool( const std::string& type,
                  const std::string& name);
  
  /// Standard Destructor
  virtual ~PrUTMagnetTool();

  LHCb::MagneticFieldGrid * m_magFieldSvc;
  /// Actual operator function
  float bdlIntegral(float ySlopeVelo,float zOrigin,float zVelo);
  float zBdlMiddle(float ySlopeVelo,float zOrigin,float zVelo);
  float dist2mom(float ySlope);
  void dxNormFactorsUT(float ySlope, std::vector<float>& nfact);
  void dxNormFactorsUT(float ySlope, std::array<float,4>& nfact);
  
  
  float zMidUT();
  float zMidField();
  float averageDist2mom();
  void prepareBdlTables();
  void prepareDeflectionTables();
  void updateField() ;

  std::vector<float> returnDxLayTable();
  std::vector<float> returnBdlTable();
  
PrTableForFunction* m_lutBdl;
protected:
  void f_bdl(float slopeY, float zOrigin , float zStart, float zStop);
  
private:
  
  float m_BdlTrack;
  float m_zHalfBdlTrack;
  std::vector<float> m_zLayers;
  
  std::vector<float> m_bdlTmp, m_zTmp;
  
  /// pointer to mag field service
  
  PrTableForFunction* m_lutZHalfBdl;
  
  float m_zCenterUT;
  float m_zMidField;
  
  float m_dist2mom;
  PrTableForFunction* m_lutDxLay;
  PrTableForFunction* m_lutDxToMom;
  
  std::vector<float> m_lutVar; // auxiliary vector used as argument
  
  DeSTDetector*       m_STDet;         ///< ST Detector element
  std::string m_detLocation;
  
  bool m_noField;
  float m_bdlIntegral_NoB;
  float m_zBdlMiddle_NoB;
  float m_zMidField_NoB;
  float m_averageDist2mom_NoB;
  
};


#endif // PRUTMAGNETTOOL_H
