#ifndef ALLEN_MUONTABLEREADER_H
#define ALLEN_MUONTABLEREADER_H

#include <array>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <fstream>
#include <ios>
#include <cassert>
#include "MuonTileID.h"

struct MuonTable {
    int gridX[16]{}, gridY[16]{};
    unsigned int offset[16]{};
    float sizeX[16]{}, sizeY[16]{};
    std::vector<std::array<float, 3>> points[4];
};

class MuonTableReader {
public:
  void read(const char* raw_input, MuonTable* pad, MuonTable* stripX, MuonTable* stripY);
};

void calcTilePos(MuonTable* pad, const MuonTileID& tile, double& x, double& deltax, double& y, double& deltay,
    double& z);

void calcStripXPos(MuonTable* stripX, MuonTileID& tile, double& x, double& deltax, double& y, double& deltay,
    double& z);

void calcStripYPos(MuonTable* stripY, MuonTileID& tile, double& x, double& deltax, double& y, double& deltay,
    double& z);

void transform_for_uncrossed_hits(MuonTileID& tile, MuonTable* pad, MuonTable* stripX, MuonTable* stripY,
    double& x, double& dx, double& y, double& dy, double& z);

#endif //ALLEN_MUONTABLEREADER_H
