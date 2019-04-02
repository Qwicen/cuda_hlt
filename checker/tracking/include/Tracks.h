/** @file Tracks.h
 *
 * @brief SOA Velo Tracks
 *
 * @author Rainer Schwemmer
 * @author Daniel Campora
 * @author Manuel Schiller
 * @date 2018-02-06
 */

#pragma once

#include <array>
#include <cstdint>

#include "LHCbID.h"

namespace trackChecker {

  class Track {

  public:
    SomeLHCbIDs allids;
    // Kalman information.
    float z, x, y, tx, ty, qop;
    float first_qop, best_qop;
    float chi2, chi2V, chi2T;
    uint ndof, ndofV, ndofT;
    float kalman_ip, kalman_ip_chi2, kalman_ipx, kalman_ipy;
    float kalman_docaz;
    float velo_ip, velo_ip_chi2, velo_ipx, velo_ipy;
    float velo_docaz;
    float long_ip, long_ip_chi2, long_ipx, long_ipy;
    std::size_t n_matched_total = 0;
    float p;
    float muon_catboost_output;
    bool is_muon;

    void addId(LHCbID id) { allids.push_back(id); }

    SomeLHCbIDs ids() const { return allids; }

    int nIDs() const { return allids.size(); }
  };

  using Tracks = std::vector<Track>;
} // namespace trackChecker
