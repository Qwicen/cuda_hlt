#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <cassert>
using namespace std;



 // structure that holds the grid parameters for one quadrant (a la DC06)
struct GridQuadrant
  {
    double zOffset ;
    double Dxyz[3] ;
    size_t Nxyz[3] ;
    //vector of vector holding field grid
    vector<vector<double>> Q;
  } ;








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




int main () {
  cout<< "hello world!"<<endl;

  // open the file
  string filename = "/home/freiss/lxplus_work/field101.c1.down.cdf";
  
  GridQuadrant quad; 
  readQuadrant(filename, quad);





    

  
  return 0;
}