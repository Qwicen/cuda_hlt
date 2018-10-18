#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "MuonDefinitions.cuh"

#define SQRT3 1.7320508075688772
#define INVSQRT3 0.5773502691896258
#define MSFACTOR 5.552176750308537


using MuonTrackExtrapolation = std::vector<std::pair<float, float>>;
//using LHCbID = int;


struct State
{
    State(float x, float y, float tx, float ty, float p) : x(x), y(y), tx(tx), ty(ty), p(p) {}
    float x;
    float y;
    float tx;
    float ty;
    float p;
    //float z;
};


std::vector<double> calcBDT(State &muTrack,
                            Muon::HitsSoA &hits);
