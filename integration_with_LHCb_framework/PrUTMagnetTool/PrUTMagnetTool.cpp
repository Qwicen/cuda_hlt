// Include files






// local

#include "PrUTMagnetTool.h"


//-----------------------------------------------------------------------------
// Implementation file for class : PrUTMagnetTool
//
// 2006-09-25 : Mariusz Witek
//-----------------------------------------------------------------------------













// Standard Constructor
PrUTMagnetTool::PrUTMagnetTool( vector<std::string> filenames)
{

  MagneticFieldGridReader magreader;

  
  m_magFieldSvc = new LHCb::MagneticFieldGrid();
  magreader.readFiles( filenames, *m_magFieldSvc);

}

//=========================================================================
//  setMidUT
//=========================================================================


//=========================================================================
// Callback function for field updates
//=========================================================================
void PrUTMagnetTool::updateField() 
{
  prepareBdlTables();
  //prepareDeflectionTables();

  m_noField = false;
  // check whether B=0
  f_bdl(0.,0.,400., m_zCenterUT);
  float  tbdl = m_BdlTrack;
  if(fabs(tbdl)<10e-4) {
    m_noField = true;
  }

  m_zMidField = zBdlMiddle(0.05, 0.0, 0.0);

  if(m_noField) {
    cout << " No B field detected." << endl;    
  }

}


//=========================================================================
// z middle of UT
//=========================================================================
float PrUTMagnetTool::zMidUT() {
  return m_zCenterUT;
}

//=========================================================================
// z middle of B field betweenz=0 and zMidUT
//=========================================================================
float PrUTMagnetTool::zMidField() {
  if( m_noField ) { 
    return m_zMidField_NoB;
  }
  return m_zMidField;
}

//=========================================================================
// averageDist2mom
//=========================================================================
float PrUTMagnetTool::averageDist2mom() {
  if( m_noField ) { 
    return m_averageDist2mom_NoB;
  }
  return m_dist2mom;
}

//=========================================================================
// prepareBdlTables
//=========================================================================
void PrUTMagnetTool::prepareBdlTables() {

  cout << "Start generation of VeloUT Bdl LUTs" << endl;
  // prepare table with Bdl integrations
  // Bdl integral depends on 3 track parameters
  //  slopeY     - y slope of the track
  //  zOrigin    - z of track intersection with z axis (in YZ projection)
  //  zVeloEnd   - z of the track at which slopeY is given
  //
  //                      slopeY    zOrigin    zVeloEnd

  // WDH: I do not understand why these tables are stored as
  // tools. what is wrong with just owning the objects?
  // m_zCenterUT is a normalization plane which should be close to middle of UT.
  // It is used to normalize dx deflection at different UT layers.
  // No need to update with small UT movement up to +- 5 cm. 

  m_zCenterUT = 2484.6;
  float zCenterUT = 0.;
  
  //manually put in z-positions for now
  /*
  m_zLayers.clear();
  for ( std::vector<DeSTLayer*>::const_iterator itL = m_STDet->layers().begin();
          m_STDet->layers().end() != itL; ++itL ) {
     float zlay = (*(*itL)->sectors().begin())->globalCentre().Z(); 
     m_zLayers.push_back(zlay); 
     zCenterUT += zlay;
  }    
  */
  zCenterUT += 2327.5;
  m_zLayers.push_back(2327.5);
  zCenterUT += 2372.5;
  m_zLayers.push_back(2372.5);
  zCenterUT += 2597.5;
  m_zLayers.push_back(2597.5);
  zCenterUT += 2642.5;
  m_zLayers.push_back(2642.5);
  zCenterUT /= m_zLayers.size();
  if (fabs( m_zCenterUT - zCenterUT ) > 50. ) {
    cout << "Calculated center of UT station far away from nominal value: " 
              << zCenterUT << " wrt nominal " << m_zCenterUT << endl;
    cout << " Calculated value taken: " << zCenterUT << endl;    
    m_zCenterUT = zCenterUT;
  }
  // warning layers not in order of increasing z
  std::sort(m_zLayers.begin(),m_zLayers.end());

  m_lutBdl      = new PrTableForFunction("PrTableForFunction/table1", "PrTableForFunction/table1");
  m_lutZHalfBdl = new PrTableForFunction("PrTableForFunction/table2", "PrTableForFunction/table1");
  m_lutBdl->clear() ;
  m_lutZHalfBdl->clear() ;

  m_lutBdl->addVariable(30, -0.3, 0.3);
  m_lutBdl->addVariable(10, -250., 250.);
  m_lutBdl->addVariable(10,    0., 800.);
  m_lutBdl->prepareTable();

  m_lutZHalfBdl->addVariable(30, -0.3, 0.3); 
  m_lutZHalfBdl->addVariable(10, -250., 250.); 
  m_lutZHalfBdl->addVariable(10,    0., 800.); 
  m_lutZHalfBdl->prepareTable(); 

  m_lutVar.clear();
  m_lutVar.push_back(0.0);
  m_lutVar.push_back(0.0);
  m_lutVar.push_back(0.0);
  m_lutBdl->resetIndexVector();
  m_lutZHalfBdl->resetIndexVector();
  int iover = 0;
  while(!iover) {
    m_lutBdl->getVariableVector(m_lutVar);
    float slopeY   = m_lutVar[0];
    float zOrigin  = m_lutVar[1];
    float zEndVelo = m_lutVar[2];
    f_bdl(slopeY, zOrigin, zEndVelo, m_zCenterUT);
    m_lutBdl->fillTable(m_BdlTrack);
    m_lutZHalfBdl->fillTable(m_zHalfBdlTrack);
    iover = m_lutBdl->incrementIndexVector();
    iover = m_lutZHalfBdl->incrementIndexVector();
  }

  cout << "Generation of VeloUT Bdl LUTs finished" << endl;
  return;

}


//=========================================================================
//  Destructor
//=========================================================================
PrUTMagnetTool::~PrUTMagnetTool() {}

//****************************************************************************
float PrUTMagnetTool::bdlIntegral(float ySlopeVelo, float zOrigin, float zVelo) {

    if( m_noField ) { 
      return m_bdlIntegral_NoB;
    }
    m_lutVar.clear();
    m_lutVar.push_back(ySlopeVelo);
    m_lutVar.push_back(zOrigin);
    m_lutVar.push_back(zVelo);
    return  m_lutBdl->getInterpolatedValueFromTable(m_lutVar);
}
//****************************************************************************
float PrUTMagnetTool::zBdlMiddle(float ySlopeVelo, float zOrigin, float zVelo) {

    if( m_noField ) { 
      return m_zBdlMiddle_NoB;
    }
    m_lutVar.clear();
    m_lutVar.push_back(ySlopeVelo);
    m_lutVar.push_back(zOrigin);
    m_lutVar.push_back(zVelo);
    return m_lutZHalfBdl->getInterpolatedValueFromTable(m_lutVar);
}

//****************************************************************************
float PrUTMagnetTool::dist2mom(float ySlope) {

  if( m_noField ) { 
    return m_averageDist2mom_NoB;
  }
  m_lutVar.clear();
  m_lutVar.push_back(ySlope);
  return m_lutDxToMom->getValueFromTable(m_lutVar);

}

//****************************************************************************
void PrUTMagnetTool::dxNormFactorsUT(float ySlope, std::vector<float>& nfact) {

  nfact.clear();
  if( m_noField ) { 
    for(int i=0; i<4; i++) {
      nfact.push_back(1.0);
    }    
    return;
  }
  m_lutVar.clear();
  m_lutVar.push_back(0.);
  m_lutVar.push_back(fabs(ySlope));
  for(int i=0; i<4; i++) {
    m_lutVar[0]=i;
    float nf = m_lutDxLay->getValueFromTable(m_lutVar);
    nfact.push_back(nf);
  }
}

//****************************************************************************
void PrUTMagnetTool::dxNormFactorsUT(float ySlope, std::array<float,4>& nfact) {

  if( m_noField ) { 
    nfact = { 1.0, 1.0, 1.0, 1.0 };
    return;
  }
  m_lutVar.clear();
  m_lutVar.push_back(0.);
  m_lutVar.push_back(fabs(ySlope));
  for(int i=0; i<4; ++i) {
    m_lutVar[0]=i;
    float nf = m_lutDxLay->getValueFromTable(m_lutVar);
    nfact[i] = nf;
  }
}


//****************************************************************************
void PrUTMagnetTool::f_bdl( float slopeY, float zOrigin,
                             float zStart, float zStop){

    m_BdlTrack=0.0;
    m_zHalfBdlTrack=0.0;

    if(zStart>zStop) return;

    float Bdl=0.0;
    float zHalfBdl=0.0;

   // vectors to calculate z of half Bdl
    m_bdlTmp.clear();
    m_zTmp.clear(); 

    // prepare m_zBdlHalf;
    XYZPoint  aPoint(0.,0.,0.);
    XYZVector bField;

    int np    = 500;
    float dz = (zStop - zStart)/np;
    float dy = dz*slopeY;

    aPoint.SetX( 0.0 );

    float z = zStart+dz/2.;
    float y = slopeY*(zStart-zOrigin);

    float bdl = 0.;


    while( z<zStop ) {
      
      aPoint.SetY( y );
      aPoint.SetZ( z );

      m_magFieldSvc->fieldVector( aPoint, bField );
      bdl += dy*bField.z() - dz*bField.y();
      if(z>100. * 10.) {
        m_bdlTmp.push_back(bdl);
        m_zTmp.push_back(z);
      }
      z += dz;
      y += dy;
    }

    Bdl=bdl;
    float bdlhalf = fabs(Bdl)/2.;

    for(unsigned int i=5; i<m_bdlTmp.size()-5; i++) {
      if(fabs(m_bdlTmp[i])>bdlhalf) {
        float zrat = (Bdl/2.-m_bdlTmp[i-1])/(m_bdlTmp[i]-m_bdlTmp[i-1]);
        zHalfBdl = m_zTmp[i-1]+dz*zrat;
        break;
      }
    } 

    m_BdlTrack = Bdl;
    m_zHalfBdlTrack = zHalfBdl;

 }

//=========================================================================
//  return the DxTable
//=========================================================================
std::vector<float> PrUTMagnetTool::returnDxLayTable() const {
  std::vector<float> deflection_vals;
  float deflection;
  ifstream deflectionfile;
  deflectionfile.open("../PrUTMagnetTool/deflection.txt");
  if (deflectionfile.is_open()) {
    while (!deflectionfile.eof()) {
      deflectionfile >> deflection;
      deflection_vals.push_back(deflection);

    }
  }

  /*
  std::cout << "here's the deflection  LUT " << std::endl;
  for (auto i = m_lutDxLay->returnTable().begin(); i != m_lutDxLay->returnTable().end(); ++i)
    std::cout << *i << ' ' << std::endl;
  std::cout << "here ends deflection LUT " << std::endl;

  return m_lutDxLay->returnTable();
  */
  std::cout << "here's the deflection  LUT " << std::endl;
  for (auto i = deflection_vals.begin(); i != deflection_vals.end(); ++i)
    std::cout << *i << ' ' << std::endl;
  std::cout << "here ends deflection LUT " << std::endl;
 return deflection_vals;
}

//=========================================================================
//  return the Bdl Table
//=========================================================================
std::vector<float> PrUTMagnetTool::returnBdlTable() const{
  std::cout << "here's the BDl LUT " << std::endl;
  for (auto i = m_lutBdl->returnTable().begin(); i != m_lutBdl->returnTable().end(); ++i)
    std::cout << *i << ' ' << std::endl;
  std::cout << "here ends BDl LUT " << std::endl;
  return m_lutBdl->returnTable();
}


//=========================================================================
// prepareDeflectionTables
//=========================================================================
void PrUTMagnetTool::prepareDeflectionTables() {

  cout << "Start generation of VeloUT deflection LUTs" << endl;


  // prepare deflection tables

  m_lutDxLay    = new PrTableForFunction("PrTableForFunction/table3","PrTableForFunction/table3");
  m_lutDxToMom  = new PrTableForFunction("PrTableForFunction/table4","PrTableForFunction/table3");
  m_lutDxLay->clear() ;
  m_lutDxToMom->clear() ;

  // Retrieve extrapolators
  float xBeg = 0.;
  float yBeg = 0.;
  float zBeg = 0.;
  float qpBeg = 1./10.;
  float dxdzBeg = 0.0;
  float dydzBeg = 0.0;



  m_lutDxLay->addVariable(3, 0., 3.); // index of UT layer 1 to 4
  m_lutDxLay->addVariable(30, 0.0, 0.3); // slope y
  m_lutDxLay->prepareTable();

  m_lutVar.clear();
  m_lutVar.push_back(0.0);
  m_lutVar.push_back(0.0);
  
  int iover = 0;
  while(!iover) {
    m_lutDxLay->getVariableVector(m_lutVar);
    int idLay       = int(m_lutVar[0]+0.000001);
    dydzBeg  = m_lutVar[1];
    dxdzBeg  = 0.0;
    float zLay = m_zLayers[idLay];
    
    xBeg = 0.;
    yBeg = 0.;
    zBeg = 0.;


    float dx_mid = 0.;
    float dx_lay = 0.;
    float ratio = 0.5;
    
    m_lutDxLay->fillTable(ratio);
    iover = m_lutDxLay->incrementIndexVector();
  }

  // distance to momentum table (depends on y slope only)
  m_lutDxToMom->addVariable(30, 0.0, 0.3); // slope y
  m_lutDxToMom->prepareTable();

  iover = 0;
  m_lutVar.clear();
  m_lutVar.push_back(0.0);
  int counter = 0;
  while(!iover) {
    m_lutDxToMom->getVariableVector(m_lutVar);
    dxdzBeg = 0.0;
    dydzBeg = m_lutVar[0];
    xBeg = 0.;
    yBeg = 0.;
    zBeg = 0.;


    float dx2mom = 0.5;
    m_lutDxToMom->fillTable(dx2mom);
    iover = m_lutDxToMom->incrementIndexVector();
    counter++;
  }
  cout<<"numbert of iterations: " << counter << endl;
  // determine distToMomentum parameter
  m_lutVar.clear();
  m_lutVar.push_back(0.05);
  m_dist2mom = m_lutDxToMom->getValueFromTable(m_lutVar);

  cout << "Generation of VeloUT deflection LUTs finished" << endl;

  return;
}


