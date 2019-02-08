#pragma once

/**
   *Constants mainly taken from PrVeloUT.h from Rec
   *  @author Mariusz Witek
   *  @date   2007-05-08
   *  @update for A-Team framework 2007-08-20 SHM
   *
   *  2017-03-01: Christoph Hasse (adapt to future framework)
   *  2018-05-05: Plácido Fernández (make standalone)
   *  2018-07:    Dorothea vom Bruch (convert to C code for GPU compatability)

 */
struct PrUTMagnetTool {
  static const int N_dxLay_vals = 124;
  static const int N_bdl_vals = 3752;

  // const float m_averageDist2mom = 0.0;
  float dxLayTable[N_dxLay_vals];
  float bdlTable[N_bdl_vals];

  PrUTMagnetTool() {}
  PrUTMagnetTool(const float* _dxLayTable, const float* _bdlTable)
  {
    for (int i = 0; i < N_dxLay_vals; ++i) {
      dxLayTable[i] = _dxLayTable[i];
    }
    for (int i = 0; i < N_bdl_vals; ++i) {
      bdlTable[i] = _bdlTable[i];
    }
  }
};
