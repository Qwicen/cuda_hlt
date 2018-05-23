
#include "MagneticFieldGridReader.h"
#include "MagneticFieldGrid.h"
#include <string>
#include <fstream>

//=============================================================================
// Read the field map files and scale by scaleFactor
//=============================================================================

////////////////////////////////////////////////////////////////////////////////////////
// structure that holds the grid parameters for one quadrant (a la DC06)
////////////////////////////////////////////////////////////////////////////////////////
struct GridQuadrant
  {
    double zOffset ;
    double Dxyz[3] ;
    size_t Nxyz[3] ;
    //vector of vector holding field grid
    vector<vector<double>> Q;
  } ;


MagneticFieldGridReader::MagneticFieldGridReader(IMessageSvc& msgsvc)
  : m_msg( &msgsvc, "MagneticFieldSvc" )
{
}


void readFiles( const std::vector<std::string>& filenames)
  //,
    //                                           LHCb::MagneticFieldGrid& grid ) 
{
  assert ( filenames.size() == 4 ) ;

  GridQuadrant quadrants[4] ;

  for ( int iquad=0; iquad<4 ; ++iquad )
  {
    readQuadrant( filenames[iquad], quadrants[iquad] ) ;
  }

  // check that the quadrants are consistent

    for ( int iquad=1; iquad<4; ++iquad ) 
    {
      assert( quadrants[0].zOffset == quadrants[iquad].zOffset ) ;
      for ( size_t icoord = 0; icoord<3; ++icoord ) 
      {
        assert( quadrants[0].Dxyz[icoord] == quadrants[iquad].Dxyz[icoord] ) ;
        assert( quadrants[0].Nxyz[icoord] == quadrants[iquad].Nxyz[icoord] ) ;
      }
    }

    // now fill the grid
    //fillGridFromQuadrants( quadrants, grid ) ;
  

  
}

StatusCode MagneticFieldGridReader::readDC06File( const std::string& filename,
                                                  LHCb::MagneticFieldGrid& grid ) const
{
  GridQuadrant quadrants[4] ;
  // read the first quadrant
  StatusCode sc = readQuadrant( filename, quadrants[0] ) ;

  if( sc.isSuccess() ) {
    // multiply by 4
    for(size_t iquad=1; iquad<4; ++iquad)
      quadrants[iquad] = quadrants[0];

    // now fill the grid
    fillGridFromQuadrants( quadrants, grid ) ;
  }

  return sc ;
}

void MagneticFieldGridReader::fillConstantField( const Gaudi::XYZVector& /* field */,
                                                 LHCb::MagneticFieldGrid& grid ) const
{
  // make a grid that spans the entire world

  // // original
  // grid.m_Dxyz[0] = 2*Gaudi::Units::km ;
  // grid.m_Dxyz[1] = 2*Gaudi::Units::km ;
  // grid.m_Dxyz[2] = 2*Gaudi::Units::km ;
  // grid.m_invDxyz[0] = 1.0 / grid.m_Dxyz[0];
  // grid.m_invDxyz[1] = 1.0 / grid.m_Dxyz[1];
  // grid.m_invDxyz[2] = 1.0 / grid.m_Dxyz[2];
  // grid.m_min_FL[0] = - Gaudi::Units::km ;
  // grid.m_min_FL[1] = - Gaudi::Units::km ;
  // grid.m_min_FL[2] = - Gaudi::Units::km ;
  // grid.m_Nxyz[0] = 2;
  // grid.m_Nxyz[1] = 2;
  // grid.m_Nxyz[2] = 2 ;
  // grid.m_Q.clear() ;
  // grid.m_Q.resize( grid.m_Nxyz[0] * grid.m_Nxyz[1] * grid.m_Nxyz[2], decltype(grid.m_Q)::value_type{field} ) ;

  // Vectorised version
  grid.m_Dxyz_V = Vec4f( 2*Gaudi::Units::km, 2*Gaudi::Units::km, 2*Gaudi::Units::km, 0. );
  grid.m_invDxyz_V = Vec4f( 1.0 / grid.m_Dxyz_V[0],
                            1.0 / grid.m_Dxyz_V[1],
                            1.0 / grid.m_Dxyz_V[2], 0.0 );
  grid.m_min_FL_V = Vec4f( -Gaudi::Units::km, -Gaudi::Units::km, -Gaudi::Units::km, 0. );
  grid.m_Nxyz_V = { 2, 2, 2 };
  grid.m_Q_V.clear() ;
  grid.m_Q_V.resize( grid.m_Nxyz_V[0] * grid.m_Nxyz_V[1] * grid.m_Nxyz_V[2], Vec4f(0,0,0,0) ) ;

}

////////////////////////////////////////////////////////////////////////////////////////
// routine to fill the grid from the 4 quadrants
////////////////////////////////////////////////////////////////////////////////////////

void MagneticFieldGridReader::fillGridFromQuadrants( GridQuadrant* quadrants,
                                                     LHCb::MagneticFieldGrid& grid ) const
{
  // flip the signs of the field such that they are all in the same frame
  for( size_t iquad=1; iquad<4; ++iquad) {
    int signx(1), signy(1), signz(1) ;
    switch(iquad) {
    case 1: signx = -1 ; break ;
    case 2: signx = -1 ; signz = -1 ; break ;
    case 3: signz = -1 ;
    }
    for( std::vector<Gaudi::XYZVector>::iterator it = quadrants[iquad].Q.begin() ;
         it != quadrants[iquad].Q.end(); ++it ) {
      *it = Gaudi::XYZVector( signx * it->x(),
                              signy * it->y(),
                              signz * it->z() ) ;
    }
  }

  // now we remap: put the 4 quadrants in a single grid
  const size_t Nxquad = quadrants[0].Nxyz[0] ;
  const size_t Nyquad = quadrants[0].Nxyz[1] ;
  const size_t Nzquad = quadrants[0].Nxyz[2] ;

  // new number of bins. take into account that they overlap at z axis
  // grid.m_Dxyz[0] = quadrants[0].Dxyz[0] ;
  // grid.m_Dxyz[1] = quadrants[0].Dxyz[1] ;
  // grid.m_Dxyz[2] = quadrants[0].Dxyz[2] ;
  grid.m_Dxyz_V = Vec4f( quadrants[0].Dxyz[0],
                         quadrants[0].Dxyz[1],
                         quadrants[0].Dxyz[2], 0.0 );
  // grid.m_invDxyz[0] = 1.0 / grid.m_Dxyz[0];
  // grid.m_invDxyz[1] = 1.0 / grid.m_Dxyz[1];
  // grid.m_invDxyz[2] = 1.0 / grid.m_Dxyz[2];
  grid.m_invDxyz_V = Vec4f( 1.0 / grid.m_Dxyz_V[0],
                            1.0 / grid.m_Dxyz_V[1],
                            1.0 / grid.m_Dxyz_V[2], 0.0 );
  // grid.m_Nxyz[0] = 2*Nxquad - 1;
  // grid.m_Nxyz[1] = 2*Nyquad - 1;
  // grid.m_Nxyz[2] = Nzquad ;
  grid.m_Nxyz_V[0] = 2*Nxquad - 1;
  grid.m_Nxyz_V[1] = 2*Nyquad - 1;
  grid.m_Nxyz_V[2] = Nzquad ;
  //grid.m_Q.resize(grid.m_Nxyz[0] * grid.m_Nxyz[1] * grid.m_Nxyz[2], {0,0,0} ) ;
  grid.m_Q_V.resize(grid.m_Nxyz_V[0] * grid.m_Nxyz_V[1] * grid.m_Nxyz_V[2], Vec4f(0,0,0,0) ) ;
  for( size_t iz=0; iz<Nzquad; ++iz)
    for( size_t iy=0; iy<Nyquad; ++iy)
      for( size_t ix=0; ix<Nxquad; ++ix)
      {

        // // 4th quadrant (negative x, negative y)
        // grid.m_Q[ grid.m_Nxyz[0] * ( grid.m_Nxyz[1]*iz + (Nyquad-iy-1) ) + (Nxquad-ix-1)] =
        //   quadrants[3].Q[Nxquad * ( Nyquad * iz + iy ) + ix ] ;
        // // 2nd quadrant (negative x, positive y)
        // grid.m_Q[ grid.m_Nxyz[0] * ( grid.m_Nxyz[1]*iz + (Nyquad+iy-1) ) + (Nxquad-ix-1)] =
        //   quadrants[1].Q[Nxquad * ( Nyquad * iz + iy ) + ix ] ;
        // // 3rd quadrant (postive x, negative y)
        // grid.m_Q[ grid.m_Nxyz[0] * ( grid.m_Nxyz[1]*iz + (Nyquad-iy-1) ) + (Nxquad+ix-1)] =
        //   quadrants[2].Q[Nxquad * ( Nyquad * iz + iy ) + ix ] ;
        // // 1st quadrant (positive x, positive y)
        // grid.m_Q[ grid.m_Nxyz[0] * ( grid.m_Nxyz[1]*iz + (Nyquad+iy-1) ) + (Nxquad+ix-1)] =
        //   quadrants[0].Q[Nxquad * ( Nyquad * iz + iy ) + ix ] ;

        // Vectorised one

        // 4th quadrant (negative x, negative y)
        const auto& Q4 = quadrants[3].Q[Nxquad * ( Nyquad * iz + iy ) + ix ];
        grid.m_Q_V[ grid.m_Nxyz_V[0] * ( grid.m_Nxyz_V[1]*iz + (Nyquad-iy-1) ) + (Nxquad-ix-1)] =
          Vec4f( Q4.x(), Q4.y(), Q4.z(), 0.0 );
        // 2nd quadrant (negative x, positive y)
        const auto& Q2 = quadrants[1].Q[Nxquad * ( Nyquad * iz + iy ) + ix ] ;
        grid.m_Q_V[ grid.m_Nxyz_V[0] * ( grid.m_Nxyz_V[1]*iz + (Nyquad+iy-1) ) + (Nxquad-ix-1)] =
          Vec4f( Q2.x(), Q2.y(), Q2.z(), 0.0 );
        // 3rd quadrant (postive x, negative y)
        const auto& Q3 = quadrants[2].Q[Nxquad * ( Nyquad * iz + iy ) + ix ] ;
        grid.m_Q_V[ grid.m_Nxyz_V[0] * ( grid.m_Nxyz_V[1]*iz + (Nyquad-iy-1) ) + (Nxquad+ix-1)] =
          Vec4f( Q3.x(), Q3.y(), Q3.z(), 0.0 );
        // 1st quadrant (positive x, positive y)
        const auto& Q1 = quadrants[0].Q[Nxquad * ( Nyquad * iz + iy ) + ix ] ;
        grid.m_Q_V[ grid.m_Nxyz_V[0] * ( grid.m_Nxyz_V[1]*iz + (Nyquad+iy-1) ) + (Nxquad+ix-1)] =
          Vec4f( Q1.x(), Q1.y(), Q1.z(), 0.0 );

      }

  // grid.m_min_FL[0] = - ((Nxquad-1) * grid.m_Dxyz[0]) ;
  // grid.m_min_FL[1] = - ((Nyquad-1) * grid.m_Dxyz[1]) ;
  // grid.m_min_FL[2] = quadrants[0].zOffset ;
  grid.m_min_FL_V  = Vec4f( - ((Nxquad-1) * grid.m_Dxyz_V[0]),
                            - ((Nyquad-1) * grid.m_Dxyz_V[1]),
                            quadrants[0].zOffset, 0 );

  if( UNLIKELY(m_msg.level() <= MSG::DEBUG) )
  {
    m_msg << MSG::DEBUG
          << "Field grid , nbins x,y,z  : (" << grid.m_Nxyz_V[0] << ","
          << grid.m_Nxyz_V[1] << "," <<  grid.m_Nxyz_V[2] << ")" << std::endl
          << "dx, xmin, xmax: "
          << "(" << grid.m_Dxyz_V[0] << "," << grid.m_min_FL_V[0] << ","
          << grid.m_min_FL_V[0] + (grid.m_Nxyz_V[0]-1) * grid.m_Dxyz_V[0] << ")" << std::endl
          << "dy, ymin, ymax: "
          << "(" << grid.m_Dxyz_V[0] << "," << grid.m_min_FL_V[1] << ","
          << grid.m_min_FL_V[1] + (grid.m_Nxyz_V[1]-1) * grid.m_Dxyz_V[1] << ")" << std::endl
          << "dz, zmin, zmax: "
          << "(" << grid.m_Dxyz_V[0] << "," << grid.m_min_FL_V[2] << ","
          << grid.m_min_FL_V[2] + (grid.m_Nxyz_V[2]-1) * grid.m_Dxyz_V[2] << ")" << endmsg ;
  }

}

////////////////////////////////////////////////////////////////////////////////////////
// read the data for a single quadrant from a file
////////////////////////////////////////////////////////////////////////////////////////

void readQuadrant( const std::string& filename, GridQuadrant& quad )  {
  std::ifstream infile( filename );

  if ( infile ) {
    cout  << "Opened magnetic field file : " << filename << endl;
        // Skip the header till PARAMETER
    char line[ 255 ];
    do{
      infile.getline( line, 255 );
    } while( line[0] != 'P' );

    // Get the PARAMETER
    std::string sPar[2];
    char* token = strtok( line, " " );
    cout << "parameter token: " << token << endl;
    int ip = 0;
    do{
      if ( token ) { sPar[ip] = token; token = strtok( nullptr, " " );}
      else continue;
      ip++;
    } while ( token != nullptr );
    long int npar = atoi( sPar[1].c_str() );
    for (int i =0 ; i < 2; i++) cout << sPar[i] << endl;
      cout << sPar[0] << endl;

    // Skip the header till GEOMETRY
    do{
      infile.getline( line, 255 );
    } while( line[0] != 'G' );

    // Skip any comment before GEOMETRY
    do{
      infile.getline( line, 255 );
    } while( line[0] != '#' );

    // Get the GEOMETRY
    infile.getline( line, 255 );
    std::string sGeom[7];
    token = strtok( line, " " );
    cout << "geometry token: " << token << endl;
    int ig = 0;
    do{
      if ( token ) { sGeom[ig] = token; token = strtok( nullptr, " " );}
      else continue;
      ig++;
    } while (token != nullptr);
    for (int i =0 ; i < 7; i++) cout << sGeom[i] << endl;
    

    // Grid dimensions are given in cm in CDF file. Convert to CLHEP units

    quad.Dxyz[0] = atof( sGeom[0].c_str() ) ;
    quad.Dxyz[1] = atof( sGeom[1].c_str() ) ;
    quad.Dxyz[2] = atof( sGeom[2].c_str() ) ;
    quad.Nxyz[0] = atoi( sGeom[3].c_str() );
    quad.Nxyz[1] = atoi( sGeom[4].c_str() );
    quad.Nxyz[2] = atoi( sGeom[5].c_str() );
    quad.zOffset = atof( sGeom[6].c_str() ) ;

    // Number of lines with data to be read
    long int nlines = ( npar - 7 ) / 3;
    

    // Skip comments and fill a vector of magnetic components for the
    // x, y and z positions given in GEOMETRY
    while( infile ) {
      // parse each line of the file,
      // comment lines begin with '#' in the cdf file
      infile.getline( line, 255 );
      if ( line[0] == '#' ) continue;
      std::string sFx, sFy, sFz;
      char* token = strtok( line, " " );
      if ( token ) { sFx = token; token = strtok( nullptr, " " );} else continue;
      if ( token ) { sFy = token; token = strtok( nullptr, " " );} else continue;
      if ( token ) { sFz = token; token = strtok( nullptr, " " );} else continue;
      if ( token != nullptr ) continue;

      // Field values are given in gauss in CDF file. Convert to CLHEP units
      double fx = std::stod( sFx ) ;
      double fy = std::stod( sFy );
      double fz = std::stod( sFz ) ;
      vector<double> test;
      test.push_back(fx);
      test.push_back(fy);
      test.push_back(fz);
      //cout << test[0] << " " << test.size() << endl;
      quad.Q.push_back(test);
      cout << quad.Q.size() <<" " << sizeof(quad.Q) <<endl;

      // Add the magnetic field components of each point
      

    }
    cout << "total size: " << sizeof(quad.Q) << endl;
    infile.close();
  }



}

