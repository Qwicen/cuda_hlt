#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <cassert>
#include "MagneticFieldGridReader.h"
#include "PrUTMagnetTool.h"
using namespace std;




int main () {

  // open the file
  string filename = "fieldmaps/field101.c1.down.cdf";
  //test magnet grid reader
  GridQuadrant quad; 
  MagneticFieldGridReader magreader;
  magreader.readQuadrant(filename, quad);
  vector<std::string> filenames;
  filenames.push_back("../PrUTMagnetTool/fieldmaps/field101.c1.down.cdf");
  filenames.push_back("../PrUTMagnetTool/fieldmaps/field101.c2.down.cdf");
  filenames.push_back("../PrUTMagnetTool/fieldmaps/field101.c3.down.cdf");
  filenames.push_back("../PrUTMagnetTool/fieldmaps/field101.c4.down.cdf");
  LHCb::MagneticFieldGrid grid;
  magreader.readFiles( filenames, grid);
  XYZPoint point (1., 1., 1.);
  grid.fieldVector(point);
  cout << grid.fieldVector(point).X() << endl;
  XYZVector bfeld;
  grid.fieldVector(point, bfeld);
  cout << grid.fieldVector(point).X() << " " << bfeld.X() << endl;
  cout << grid.fieldVector(point).Y() << " " << bfeld.Y() << endl;
  cout << grid.fieldVector(point).Z() << " " << bfeld.Z() << endl;


  //test standalone magnet tool
  PrUTMagnetTool testtool(filenames);

  //prepare Bdl LUT
  testtool.prepareBdlTables();
  testtool.returnBdlTable();

  // prepare deflection LUT
  // still dummy implementation -> need extrapolators
  testtool.prepareDeflectionTables() ;
  testtool.returnDxLayTable();


  testtool.zMidUT();
  testtool.averageDist2mom();


  testtool.m_magFieldSvc->fieldVector(point, bfeld);
   cout << grid.fieldVector(point).X() << " " << bfeld.X() << endl;
  cout << grid.fieldVector(point).Y() << " " << bfeld.Y() << endl;
  cout << grid.fieldVector(point).Z() << " " << bfeld.Z() << endl;
    

  
  return 0;
}