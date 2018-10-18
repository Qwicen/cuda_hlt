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

struct Track
{
    Track()
    {
        p = float(rand());
    }
    State closestState(float f)
    {
        return *new State(1,2,3,4,5);
    };
    
    float p;
};


struct CommonMuonTool
{
    CommonMuonTool()
    {
        for (auto i = 0; i < Muon::Constants::n_stations; ++i)
        {
            m_stationZ.push_back(float(rand()));
        }
    }
    using MuonTrackExtrapolation = std::vector<std::pair<float, float>>;
    std::vector<float> m_stationZ;
    size_t m_stationsCount = Muon::Constants::n_stations;
    
    MuonTrackExtrapolation extrapolateTrack(Track& track)
    {
        MuonTrackExtrapolation extrapolation;
        State state = track.closestState(0);
        for (unsigned station = 0; station != m_stationsCount; ++station) {
            extrapolation.emplace_back(state.x + state.tx * (m_stationZ[station] ),
                                       state.y + state.ty * (m_stationZ[station] ));
        }
        return extrapolation;
    }
};


std::vector<double> calcBDT(State &muTrack,
                            Muon::HitsSoA &hits);
