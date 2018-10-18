#include "MuonFeatures.h"


std::vector<double> calcBDT(State &muTrack,
                            Muon::HitsSoA &hits)
{
    // features
    std::vector<double> times, dts, cross, resX, resY, minDist, distSeedHit;
    
    // let's start
    std::vector<float>m_stationZ;
    std::vector<std::pair<float,float>> extrapolation;
    for (int i_station = 0; i_station < Muon::Constants::n_stations; ++i_station) {
	const int station_offset = hits.m_station_offsets[i_station];
	m_stationZ.push_back(hits.m_z[station_offset]);
	extrapolation.emplace_back(muTrack.x + muTrack.tx * m_stationZ[i_station],
			           muTrack.y + muTrack.ty * m_stationZ[i_station]);
    }
    
    for( unsigned int st = 0; st != Muon::Constants::n_stations; ++st ){
        times.push_back(-10000.);
        dts.push_back(-10000.);
        cross.push_back(0.);
        resX.push_back(-10000.);
        resY.push_back(-10000.);
        minDist.push_back(1e10);
        distSeedHit.push_back(1e6);
    }

    std::vector<int> closestHits(Muon::Constants::n_stations);
    unsigned s = 0;
    for (int i_station = 0; i_station < Muon::Constants::n_stations; ++i_station) {
	const int station_offset = hits.m_station_offsets[i_station];
        const int number_of_hits = hits.m_number_of_hits_per_station[i_station];
	for(int i_hit = 0; i_hit < number_of_hits; ++i_hit) {
            const int idx = station_offset + i_hit;
            const int id = hits.m_tile[idx];
            s = i_station;
            distSeedHit[s] = (hits.m_x[idx] - extrapolation[s].first)*(hits.m_x[idx] - extrapolation[s].first) + (hits.m_y[idx] - extrapolation[s].second)*(hits.m_y[idx] - extrapolation[s].second);
            if(distSeedHit[s] < minDist[s]) {
                minDist[s] = distSeedHit[s];
                closestHits[s] = id;
            }
        }
    };
    
    float commonFactor = MSFACTOR/muTrack.p;
    for( unsigned int st = 0; st != Muon::Constants::n_stations; ++st ){
        unsigned s = 0;
        int idFromTrack = closestHits[st];
        for (int i_station = 0; i_station < Muon::Constants::n_stations; ++i_station) {
	    const int station_offset = hits.m_station_offsets[i_station];
            const int number_of_hits = hits.m_number_of_hits_per_station[i_station];
	    for(int i_hit = 0; i_hit < number_of_hits; ++i_hit) {
                const int idx = station_offset + i_hit;
                int idFromHit = hits.m_tile[idx];
                if (idFromHit == idFromTrack) {
                    s = i_station;
                    times[s] = hits.m_time[idx];
                    dts[s] = hits.m_delta_time[idx];
                    (hits.m_uncrossed[idx]==0) ? cross[s] = 2. : cross[s] = hits.m_uncrossed[idx];
                    float travDist = sqrt((m_stationZ[s]-m_stationZ[0])*(m_stationZ[s]-m_stationZ[0])+
                                      (extrapolation[s].first-extrapolation[0].first)*(extrapolation[s].first-extrapolation[0].first)+
                                      (extrapolation[s].second-extrapolation[0].second)*(extrapolation[s].second-extrapolation[0].second));
                    float errMS = commonFactor*travDist*sqrt(travDist)*0.23850119787527452;
                    if(std::abs(extrapolation[s].first-hits.m_x[idx])!=2000){
                        resX[s] = (extrapolation[s].first-hits.m_x[idx])/sqrt((hits.m_dx[idx]*INVSQRT3)*(hits.m_dx[idx]*INVSQRT3)+errMS*errMS);
                    }
                    if(std::abs(extrapolation[s].second-hits.m_y[idx])!=2000){
                        resY[s] = (extrapolation[s].second-hits.m_y[idx])/sqrt((hits.m_dy[idx]*INVSQRT3)*(hits.m_dy[idx]*INVSQRT3)+errMS*errMS);
                    }
                }
            }
        };
    }
    
    
    std::vector<double> Input = {dts[0],dts[1],dts[2],dts[3],
        times[0],times[1],times[2],times[3],
        cross[0],cross[1],cross[2],cross[3],
        resX[0],resX[1],resX[2],resX[3],
        resY[0],resY[1],resY[2],resY[3]};

    return Input;
}

