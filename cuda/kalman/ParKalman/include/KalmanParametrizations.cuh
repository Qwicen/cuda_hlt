#pragma once

#include "Common.h"
#include "KalmanParametrizationsCoef.cuh"
#include "ParKalmanDefinitions.cuh"
#include "ParKalmanMath.cuh"

#include <fstream>
#include <sstream>

namespace ParKalmanFilter {

  //----------------------------------------------------------------------
  // Class for storing polarity.
  enum class Polarity { Up, Down };

  //----------------------------------------------------------------------
  // Structure for storing parameters and performing extrapolations.
  struct KalmanParametrizations {

    // Parameters.
    StandardCoefs C[nBinXMax][nBinYMax];

    double Par_predictV[nSetsV][nParsV];
    double Par_predictVUT[nSetsVUT][nParsVUT];
    double Par_predictUT[nSetsUT][nParsUT];
    double Par_predictUTFUT[nSetsUTFUT][nParsUTFUT];
    double Par_predictUTTF[nSetsUTTF][nParsUTTF];
    double Par_predictTFT[nSetsTFT][nParsTFT];
    double Par_predictT[nSetsT][nParsT];
    double Par_TLayer[nSetsTLayer][nParsTLayer];
    double Par_UTLayer[nSetsUTLayer][nParsUTLayer];

    double ZINI, ZFIN;
    double PMIN;
    double BENDX, BENDX_X2, BENDX_Y2, BENDY_XY;
    double Txmax, Tymax, XFmax, Xmax, Ymax;
    double Dtxy;
    double step;

    // Parameters that change sign under parity flip.
    const int flip_Par_predictV[1] = {4};
    const int flip_Par_predictVUT[4] = {0, 8, 9, 10};
    const int flip_Par_predictUT[4] = {5, 6, 7, 10};
    const int flip_Par_predictUTFUT[1] = {0};
    const int flip_Par_predictUTTF[8] = {0, 2, 3, 5, 6, 8, 9, 11};
    const int flip_Par_predictTFT[2] = {5, 6};
    const int flip_Par_predictT[3] = {5, 6, 7};

// Parameters for Pierre's UT-T extrapolation.
#define QUADRATICINTERPOLATION 1
    int Nbinx, Nbiny;
    int XGridOption, YGridOption;
    int DEGX1, DEGX2, DEGY1, DEGY2;

    // Keep track of polarity and whether or not parameters have been loaded.
    Polarity m_Polarity = Polarity::Up;
    bool paramsLoaded = false;
    bool m_qop_flip = false;

    __device__ __host__ double UTTExtrEndZ() const;
    __device__ __host__ double UTTExtrBeginZ() const;
    __device__ __host__ double VUTExtrEndZ() const;

    // Pierre's extrapolation.
    __host__ void read_params_UTT(std::string file);

    //----------------------------------------------------------------------
    // Template function for reading parameters.
    template<int nPars, int nSets>
    __host__ void read_params(std::string file, double (&params)[nSets][nPars])
    {

      // Set all parameters to 0.
      for (int i = 0; i < nSets; i++)
        for (int j = 0; j < nPars; j++)
          params[i][j] = 0.;

      // Setup infile stream.
      std::string line;
      std::ifstream myFile(file);

      // Read parameters.
      bool foundSet = false;
      if (myFile.is_open()) {
        int iSet = 0;
        while (getline(myFile, line)) {
          // determine which parameterset the respective line of paramters belongs to
          foundSet = false;
          for (int s = 0; s < nSets; s++) {
            std::stringstream ss;
            ss << "_" << s << "_";
            std::string str = ss.str();
            if (line.find(str) != std::string::npos) {
              iSet = s;
              foundSet = true;
            }
          }
          if (!foundSet) continue;
          // set the values
          std::istringstream iss(line);
          std::string sub;
          iss >> sub;
          int p = 0;
          while (iss >> sub && p < nPars) {
            params[iSet][p] = std::atof(sub.c_str());
            p++;
          }
        }
        myFile.close();
      }
      else {
        throw StrException("Failed to set the KalmanParametrizations parameters from file " + file);
      }
    }

    //----------------------------------------------------------------------
    // Flip parameters if necessary.
    template<int nPars, int nSets, int nFlips>
    __host__ void SwitchParamsForPolarity(double (&params)[nSets][nPars], const int (&flips)[nFlips])
    {
      for (int i = 0; i < nSets; i++)
        for (int j = 0; j < nFlips; j++)
          params[i][flips[j]] *= -1;
    }

    //----------------------------------------------------------------------
    // Set parameters.
    __host__ void SetParameters(std::string ParamFileLocation, Polarity polarity, bool useOneParameterSet = false)
    {

      // Get polarity.
      if ((m_Polarity == polarity) && paramsLoaded) return;
      std::string pol = (polarity == Polarity::Up ? "Up" : "Down");

      // The down parameter set is the default one
      if (useOneParameterSet) pol = "Down";
      std::string parameterPath = ParamFileLocation + "/Mag" + pol;

      // read the parameters for parametrizations
      read_params(parameterPath + "/params_predictV.txt", Par_predictV);
      read_params(parameterPath + "/params_predictVUT.txt", Par_predictVUT);
      read_params(parameterPath + "/params_predictUT.txt", Par_predictUT);
      read_params(parameterPath + "/params_predictUTFUT.txt", Par_predictUTFUT);
      read_params(parameterPath + "/params_predictUTTF.txt", Par_predictUTTF);
      read_params(parameterPath + "/params_predictTFT.txt", Par_predictTFT);
      read_params(parameterPath + "/params_predictT.txt", Par_predictT);
      read_params(parameterPath + "/params_TLayer.txt", Par_TLayer);
      read_params(parameterPath + "/params_UTLayer.txt", Par_UTLayer);

      // Get the up parameters from the down parameters
      if (useOneParameterSet && polarity == Polarity::Up) {
        SwitchParamsForPolarity(Par_predictV, flip_Par_predictV);
        SwitchParamsForPolarity(Par_predictVUT, flip_Par_predictVUT);
        SwitchParamsForPolarity(Par_predictUT, flip_Par_predictUT);
        SwitchParamsForPolarity(Par_predictUTFUT, flip_Par_predictUTFUT);
        SwitchParamsForPolarity(Par_predictUTTF, flip_Par_predictUTTF);
        SwitchParamsForPolarity(Par_predictTFT, flip_Par_predictTFT);
        SwitchParamsForPolarity(Par_predictT, flip_Par_predictT);
      }

      // read_params_UTT(ParamFileLocation + "/MagDown/v5r0_7957.tab");
      read_params_UTT(parameterPath + "/v5r0_7957.tab");
      if (polarity == Polarity::Up) m_qop_flip = true;

      m_Polarity = polarity;
      paramsLoaded = true;
    }
  };

} // namespace ParKalmanFilter
