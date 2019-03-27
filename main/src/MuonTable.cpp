#include "MuonTable.h"

struct MuonTable {
    int gridX[16]{}, gridY[16]{};
    unsigned int offset[16]{};
    float sizeX[16]{}, sizeY[16]{};
    vector<array<float, 3>> points[4];
};

class MuonTableReader {
public:
  void read(const char* raw_input) {
    MuonTable pad = MuonTable();
    MuonTable stripX = MuonTable();
    MuonTable stripY = MuonTable();
    MuonTable *muonTables[3] = {&pad, &stripX, &stripY};
    for (MuonTable* muonTable : muonTables) {
      size_t gridXSize;
      std::copy_n((size_t*) raw_input, 1, &gridXSize);
      assert(gridXSize == 16);
      raw_input += sizeof(size_t);
      std::copy_n((int*) raw_input, gridXSize, muonTable -> gridX);
      raw_input += sizeof(int) * gridXSize;

      size_t gridYSize;
      std::copy_n((size_t*) raw_input, 1, &gridYSize);
      assert(gridYSize == 16);
      raw_input += sizeof(size_t);
      std::copy_n((int*) raw_input, gridYSize, muonTable -> gridY);
      raw_input += sizeof(int) * gridYSize;

      size_t sizeXSize;
      std::copy_n((size_t *) raw_input, 1, &sizeXSize);
      assert(sizeXSize == 16);
      raw_input += sizeof(size_t);
      std::copy_n((float*) raw_input, sizeXSize, muonTable -> sizeX);
      raw_input += sizeof(float) * sizeXSize;

      size_t sizeYSize;
      std::copy_n((size_t*) raw_input, 1, &sizeYSize);
      assert(sizeYSize == 16);
      raw_input += sizeof(size_t);
      std::copy_n((float*) raw_input, sizeYSize, muonTable -> sizeY);
      raw_input += sizeof(float) * sizeYSize;

      size_t offsetSize;
      std::copy_n((size_t*) raw_input, 1, &offsetSize);
      assert(offsetSize == 16);
      raw_input += sizeof(size_t);
      std::copy_n((unsigned int*) raw_input, offsetSize, muonTable -> offset);
      raw_input += sizeof(unsigned int) * offsetSize;

      size_t tableSize;
      std::copy_n((size_t*) raw_input, 1, &tableSize);
      raw_input += sizeof(size_t);
      cout << tableSize << "\n";
      assert(tableSize == 4);
      for (int i = 0; i < tableSize; i++) {
        size_t stationTableSize;
        std::copy_n((size_t*) raw_input, 1, &stationTableSize);
        raw_input += sizeof(size_t);
        (muonTable -> points)[i].resize(stationTableSize);
        cout << stationTableSize << "\n";
        for (int j = 0; j < stationTableSize; j++) {
          float point[3];
          std::copy_n((float *) raw_input, 3, point);
          raw_input += sizeof(float) * 3;
          for (int k = 0; k < 3; k++) {
            (muonTable->points)[i][j][k] = point[k];
          }
        }
      }
    }
  }
};

void transform_for_uncrossed_hits(MuonCoords* muonCoords) {

  double x, dx, y, dy, z, dz;
  if ( coord.key().station() > ( m_nStations - 3 ) && coord.key().region() == 0 ) {
    calcTilePos( coord.key(), x, dx, y, dy, z, dz );
  } else {
    if ( coord.key().layout() == layoutOne ) {
      calcStripXPos( coord.key(), x, dx, y, dy, z, dz );
    } else {
      calcStripYPos( coord.key(), x, dx, y, dy, z, dz );
    }
  }
}

class MuonTileID {

};

constexpr unsigned int padGridX[4]{48, 48, 12, 12};
constexpr unsigned int padGridY[4]{8, 8, 8, 8};
constexpr array<int, 16> stripXGridX{48, 48, 48, 48, 48, 48, 48, 48, 12, 12, 12, 12, 12, 12, 12, 12};
constexpr array<int, 16> stripXGridY{1, 2, 2, 2, 1, 2, 2, 2, 8, 2, 2, 2, 8, 2, 2, 2};
constexpr array<int, 16> stripYGridX{8, 4, 2, 2, 8, 4, 2, 2, 12, 4, 2, 2, 12, 4, 2, 2};
constexpr array<int, 16> stripYGridY{8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};

void calcTilePos(MuonTable* pad, const MuonTileID& tile, double& x,
    double& deltax, double& y, double& deltay, double& z, double& deltaz ) const {
  int          station    = tile.station();
  int          region     = tile.region();
  int          quarter    = tile.quarter();
  int          perQuarter = 3 * padGridX[station] * padGridY[station];
  unsigned int xpad       = tile.nX();
  unsigned int ypad       = tile.nY();
  unsigned int index      = ( region * 4 + quarter ) * perQuarter; //???

  if ( ypad < padGridY[station] ) {
    index = index + padGridX[station] * ypad + xpad - padGridX[station];
  } else {
    index = index + padGridX[station] * padGridY[station] +
            2 * padGridX[station] * ( ypad - padGridY[station] ) + xpad;
  }

  auto& p = (pad -> points)[station][index];
  x                        = p[0];
  y                        = p[1];
  z                        = p[2];
  deltax                   = (pad -> sizeX)[station * 4 + region];
  deltay                   = (pad -> sizeY)[station * 4 + region];
}


void calcStripXPos(MuonTable* stripX, MuonTileID& tile, double& x, double& deltax, double& y,
                                           double& deltay, double& z, double& /*deltaz*/ ) const {

  int station        = tile.station();
  int region         = tile.region();
  int quarter        = tile.quarter();
  int perQuarter     = 3 * stripXGridX[station * 4 + region] * stripXGridY[station * 4 + region];
  unsigned int xpad  = tile.nX();
  unsigned int ypad  = tile.nY();
  unsigned int index = m_stripXOffset[station * 4 + region] + quarter * perQuarter; //???

  if ( ypad < stripXGridY[station * 4 + region] ) {
    index = index + stripXGridX[station * 4 + region] * ypad + xpad -
            stripXGridX[station * 4 + region];
  } else {
    index = index + stripXGridX[station * 4 + region] * stripXGridY[station * 4 + region] +
            2 * stripXGridX[station * 4 + region] * ( ypad - stripXGridY[station * 4 + region] ) +
            xpad;
  }

  auto& p = (stripX -> points)[station][index];
  x                        = p[0];
  y                        = p[1];
  z                        = p[2];
  deltax                   = (stripX -> sizeX)[station * 4 + region];
  deltay                   = (stripX -> sizeY)[station * 4 + region];
}

void calcStripYPos(MuonTable* stripY, MuonTileID& tile, double& x, double& deltax, double& y,
                                           double& deltay, double& z, double& /*deltaz*/ ) const {

  int station        = tile.station();
  int region         = tile.region();
  int quarter        = tile.quarter();
  int perQuarter     = 3 * stripYGridX[station * 4 + region] * stripYGridY[station * 4 + region];
  unsigned int xpad  = tile.nX();
  unsigned int ypad  = tile.nY();
  unsigned int index = m_stripYOffset[station * 4 + region] + quarter * perQuarter; //???

  if ( ypad < stripYGridY[station * 4 + region] ) {
    index = index + stripYGridX[station * 4 + region] * ypad + xpad -
            stripYGridX[station * 4 + region];
  } else {
    index = index + stripYGridX[station * 4 + region] * stripYGridY[station * 4 + region] +
            2 * stripYGridX[station * 4 + region] * ( ypad - stripYGridY[station * 4 + region] ) +
            xpad;
  }

  auto& p = (stripY -> points)[station][index];
  x                        = p[0];
  y                        = p[1];
  z                        = p[2];
  deltax                   = (stripY -> sizeX)[station * 4 + region];
  deltay                   = (stripY -> sizeY)[station * 4 + region];
}

char raw_input[1200000];
int main() {

  ifstream input("muon_table.bin", ios::binary);
  //ifstream input("a.txt");
  input.read(raw_input, 1200000);
  /*for (int i = 0; i < 3000; i++) {
    std::cout << raw_input[i] << " ";
  }*/
  std::cout << "\n";
  input.close();
  auto muonTableReader = MuonTableReader();
	muonTableReader.read(raw_input);
}